"""Streamlit interface for running the original KGEC pipeline.

This app intentionally mirrors the command-line scripts distributed with the
repository. The goal is to expose the same configuration options and produce
identical calibration metrics to the original implementations (e.g.
``Main-RotatE.py``). Only this file has been modified so that the Streamlit
demo faithfully executes the upstream training and calibration routine.
"""

from __future__ import annotations

import gc
import importlib.util
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping

import numpy as np
import pandas as pd
import streamlit as st
import torch

from dataloader import BidirectionalOneShotIterator, TrainDataset
from KGEC_method import KGEC, UncalCalibrator


# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="KGE Calibrator Demo", layout="wide")


# ---------------------------------------------------------------------------
# Utilities for loading the original main scripts dynamically
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_session_directory() -> Path:
    """Return a unique temporary directory for the current Streamlit session."""

    session_key = "kge_session_dir"
    if session_key not in st.session_state:
        temp_dir = Path(tempfile.mkdtemp(prefix="kge_calibrator_"))
        st.session_state[session_key] = str(temp_dir)
    return Path(st.session_state[session_key])


def _extract_uploaded_archive(upload: "st.runtime.uploaded_file_manager.UploadedFile", label: str) -> Path:
    """Persist ``upload`` to disk and extract it into a dedicated subdirectory."""

    session_dir = _ensure_session_directory()
    target_root = session_dir / label
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    archive_path = target_root.with_suffix(".zip")
    with archive_path.open("wb") as fout:
        fout.write(upload.getbuffer())

    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(target_root)

    # Some archives wrap the contents in a top-level directory. If exactly one
    # directory was extracted, treat it as the dataset/checkpoint root.
    children = [child for child in target_root.iterdir() if child.is_dir() and child.name != "__MACOSX"]
    files = [child for child in target_root.iterdir() if child.is_file()]
    if len(children) == 1 and not files:
        target_root = children[0]

    return target_root


def _persist_uploaded_file(upload: "st.runtime.uploaded_file_manager.UploadedFile", filename: str) -> Path:
    """Persist an uploaded file into the session directory."""

    session_dir = _ensure_session_directory()
    target_path = session_dir / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as fout:
        fout.write(upload.getbuffer())
    return target_path


def _default_output_directory() -> str:
    session_dir = _ensure_session_directory()
    output_dir = session_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


MAIN_SCRIPTS: Mapping[str, str] = {
    "RotatE": "Main-RotatE.py",
    "TransE": "Main-TransE.py",
    "DistMult": "Main-DistMult.py",
    "ComplEx": "Main-ComplEx.py",
}


class ModuleLoadError(RuntimeError):
    """Raised when one of the training scripts cannot be imported."""


@st.cache_resource(show_spinner=False)
def load_main_module(script_name: str):
    """Load a ``Main-*.py`` script as a Python module."""

    script_path = REPO_ROOT / script_name
    if not script_path.exists():
        raise ModuleLoadError(f"Script '{script_name}' not found under {REPO_ROOT}.")

    module_name = f"kge_main_{script_path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ModuleLoadError(f"Unable to create module spec for '{script_name}'.")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - import errors bubble to UI
        raise ModuleLoadError(f"Failed to import '{script_name}': {exc}") from exc

    return module


def _default_args(module) -> MutableMapping[str, object]:
    """Return a mutable copy of the argparse namespace from the script."""

    args = module.parse_args([])
    return vars(args).copy()


def _ensure_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


@dataclass
class PipelineResult:
    """Collects artefacts generated by the execution of the pipeline."""

    training_history: List[Mapping[str, float]]
    validation_history: List[Mapping[str, float]]
    validation_final: Mapping[str, float]
    test_original: Mapping[str, float]
    test_calibrated: Mapping[str, float]
    log_text: str


def _apply_config_overrides_from_file(args_namespace, config_path: Path) -> None:
    """Apply configuration overrides from ``config.json`` when no checkpoint is available."""

    if not config_path.exists():
        raise ValueError(f"Configuration file not found at {config_path}.")

    with config_path.open() as fjson:
        argparse_dict = json.load(fjson)

    if not getattr(args_namespace, "data_path", None):
        args_namespace.data_path = argparse_dict.get("data_path")

    for key in (
        "model",
        "double_entity_embedding",
        "double_relation_embedding",
        "hidden_dim",
        "test_batch_size",
    ):
        if key in argparse_dict:
            setattr(args_namespace, key, argparse_dict[key])


def execute_pipeline(module, args_dict: Mapping[str, object]) -> PipelineResult:
    """Execute the original training & calibration loop using ``args_dict``."""

    args_namespace = module.parse_args([])
    for key, value in args_dict.items():
        setattr(args_namespace, key, value)

    if not (args_namespace.do_train or args_namespace.do_valid or args_namespace.do_test):
        raise ValueError("At least one of training, validation, or testing must be enabled.")

    config_path_value = getattr(args_namespace, "config_path", None)
    entity_embedding_path_value = getattr(args_namespace, "entity_embedding_path", None)
    relation_embedding_path_value = getattr(args_namespace, "relation_embedding_path", None)

    if getattr(args_namespace, "init_checkpoint", None):
        module.override_config(args_namespace)
    elif config_path_value:
        _apply_config_overrides_from_file(args_namespace, Path(config_path_value))
    elif args_namespace.data_path is None:
        raise ValueError("Either an initial checkpoint or a data path must be provided.")

    if args_namespace.do_train and args_namespace.save_path is None:
        raise ValueError("A save path is required when training is enabled.")

    if args_namespace.save_path:
        _ensure_directory(args_namespace.save_path)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

    module.set_logger(args_namespace)
    log_buffer = io.StringIO()
    stream_handler = logging.StreamHandler(log_buffer)
    stream_handler.setLevel(logging.INFO)
    logging.getLogger("").addHandler(stream_handler)

    training_history: List[Mapping[str, float]] = []
    validation_history: List[Mapping[str, float]] = []
    validation_final: Mapping[str, float] = {}

    from calibration_training import KGEModel
    from torch.utils.data import DataLoader

    try:
        data_path = Path(args_namespace.data_path)
        with (data_path / "entities.dict").open() as fin:
            entity2id = {entity: int(eid) for eid, entity in
                         (line.strip().split("\t") for line in fin)}

        with (data_path / "relations.dict").open() as fin:
            relation2id = {relation: int(rid) for rid, relation in
                           (line.strip().split("\t") for line in fin)}

        args_namespace.nentity = len(entity2id)
        args_namespace.nrelation = len(relation2id)

        train_triples = module.read_triple(str(data_path / "train.txt"), entity2id, relation2id)
        valid_triples = module.read_triple(str(data_path / "valid.txt"), entity2id, relation2id)
        test_triples = module.read_triple(str(data_path / "test.txt"), entity2id, relation2id)
        all_true_triples = train_triples + valid_triples + test_triples

        kge_model = KGEModel(
            model_name=args_namespace.model,
            nentity=args_namespace.nentity,
            nrelation=args_namespace.nrelation,
            hidden_dim=args_namespace.hidden_dim,
            gamma=args_namespace.gamma,
            double_entity_embedding=args_namespace.double_entity_embedding,
            double_relation_embedding=args_namespace.double_relation_embedding,
        )

        if args_namespace.cuda and torch.cuda.is_available():
            device = torch.device(args_namespace.cuda_device)
            kge_model = kge_model.to(device)
        else:
            args_namespace.cuda = False
            device = torch.device("cpu")

        if args_namespace.do_train:
            train_dataloader_head = DataLoader(
                TrainDataset(
                    train_triples,
                    args_namespace.nentity,
                    args_namespace.nrelation,
                    args_namespace.negative_sample_size,
                    "head-batch",
                ),
                batch_size=args_namespace.batch_size,
                shuffle=True,
                num_workers=max(1, args_namespace.cpu_num // 2),
                collate_fn=TrainDataset.collate_fn,
            )

            train_dataloader_tail = DataLoader(
                TrainDataset(
                    train_triples,
                    args_namespace.nentity,
                    args_namespace.nrelation,
                    args_namespace.negative_sample_size,
                    "tail-batch",
                ),
                batch_size=args_namespace.batch_size,
                shuffle=True,
                num_workers=max(1, args_namespace.cpu_num // 2),
                collate_fn=TrainDataset.collate_fn,
            )

            train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

            current_learning_rate = args_namespace.learning_rate
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate,
            )
            warm_up_steps = args_namespace.warm_up_steps or args_namespace.max_steps // 2

        if getattr(args_namespace, "init_checkpoint", None):
            checkpoint_path = Path(args_namespace.init_checkpoint) / "checkpoint"
            checkpoint = torch.load(checkpoint_path, map_location=args_namespace.cuda_device if args_namespace.cuda else "cpu")
            init_step = checkpoint["step"]
            kge_model.load_state_dict(checkpoint["model_state_dict"])
            if args_namespace.do_train:
                current_learning_rate = checkpoint["current_learning_rate"]
                warm_up_steps = checkpoint["warm_up_steps"]
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif entity_embedding_path_value and relation_embedding_path_value:
            entity_embedding_path = Path(entity_embedding_path_value)
            relation_embedding_path = Path(relation_embedding_path_value)
            entity_array = np.load(entity_embedding_path)
            relation_array = np.load(relation_embedding_path)

            if tuple(entity_array.shape) != tuple(kge_model.entity_embedding.shape):
                raise ValueError(
                    "Entity embedding shape %s does not match model expectations %s."
                    % (entity_array.shape, tuple(kge_model.entity_embedding.shape))
                )
            if tuple(relation_array.shape) != tuple(kge_model.relation_embedding.shape):
                raise ValueError(
                    "Relation embedding shape %s does not match model expectations %s."
                    % (relation_array.shape, tuple(kge_model.relation_embedding.shape))
                )

            entity_tensor = torch.from_numpy(entity_array).to(
                device=device,
                dtype=kge_model.entity_embedding.dtype,
            )
            relation_tensor = torch.from_numpy(relation_array).to(
                device=device,
                dtype=kge_model.relation_embedding.dtype,
            )

            kge_model.entity_embedding.data.copy_(entity_tensor)
            kge_model.relation_embedding.data.copy_(relation_tensor)
            init_step = 0
        else:
            init_step = 0

        step = init_step
        calibration_models_list: List = []

        if args_namespace.do_train:
            training_logs: List[Mapping[str, float]] = []
            for step in range(init_step, args_namespace.max_steps):
                log = kge_model.train_step(kge_model, optimizer, train_iterator, args_namespace)
                training_logs.append(log)

                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, kge_model.parameters()),
                        lr=current_learning_rate,
                    )
                    warm_up_steps = warm_up_steps * 3

                if step % args_namespace.save_checkpoint_steps == 0:
                    save_vars = {
                        "step": step,
                        "current_learning_rate": current_learning_rate,
                        "warm_up_steps": warm_up_steps,
                    }
                    module.save_model(kge_model, optimizer, save_vars, args_namespace)

                if step % args_namespace.log_steps == 0:
                    metrics = {
                        metric: float(np.mean([entry[metric] for entry in training_logs]))
                        for metric in training_logs[0]
                    }
                    training_history.append({"step": float(step), **metrics})
                    module.log_metrics("Training average", step, metrics)
                    training_logs = []

                if args_namespace.do_valid and step % args_namespace.valid_steps == 0:
                    metrics = kge_model.test_step(
                        kge_model,
                        valid_triples,
                        all_true_triples,
                        args_namespace,
                        calibration_models_list,
                    )
                    validation_history.append({"step": float(step), **{k: float(v) for k, v in metrics.items()}})
                    module.log_metrics("Valid", step, metrics)

            save_vars = {
                "step": step,
                "current_learning_rate": current_learning_rate,
                "warm_up_steps": warm_up_steps,
            }
            module.save_model(kge_model, optimizer, save_vars, args_namespace)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        calibration_models_list.append(UncalCalibrator())
        calibration_models_list.append(
            KGEC(
                num_bins=args_namespace.KGEC_num_bins,
                init_temp=args_namespace.KGEC_initial_temperature,
                lr=args_namespace.KGEC_learning_rate,
            )
        )

        metrics = kge_model.test_step(
            kge_model,
            valid_triples,
            all_true_triples,
            args_namespace,
            calibration_models_list,
            True,
        )
        validation_final = {k: float(v) for k, v in metrics.items()}

        test_original, test_calibrated = kge_model.calibration_predict(
            kge_model,
            test_triples,
            all_true_triples,
            args_namespace,
            calibration_models_list,
        )

        return PipelineResult(
            training_history=training_history,
            validation_history=validation_history,
            validation_final=validation_final,
            test_original={k: float(v) for k, v in test_original.items()},
            test_calibrated={k: float(v) for k, v in test_calibrated.items()},
            log_text=log_buffer.getvalue(),
        )

    finally:
        logging.getLogger("").removeHandler(stream_handler)


# ---------------------------------------------------------------------------
# Streamlit layout helpers
# ---------------------------------------------------------------------------

def _format_metric_table(metrics: Mapping[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})


def _split_calibration_metrics(metrics: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, float]] = {}
    for key, value in metrics.items():
        if "_" not in key:
            grouped.setdefault("Combined", {})[key] = value
            continue
        calibrator, metric = key.split("_", 1)
        grouped.setdefault(calibrator, {})[metric] = value
    return grouped


def _display_training_history(history: Iterable[Mapping[str, float]], title: str) -> None:
    history = list(history)
    if not history:
        return
    st.subheader(title)
    df = pd.DataFrame(history)
    numeric_cols = [col for col in df.columns if col != "step"]
    st.dataframe(df.style.format({col: "{:.6f}" for col in numeric_cols}), use_container_width=True)


def _display_logs(log_text: str) -> None:
    if not log_text.strip():
        return
    st.subheader("Execution Log")
    st.text(log_text)


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

st.title("KGE Calibrator Demo")
st.markdown(
    """
    This Streamlit application runs the **exact** KGEC training and evaluation
    pipeline distributed with the repository. Configure the options in the
    sidebar and execute the routine to reproduce the original command-line
    results.
    """
)

with st.sidebar:
    st.header("Configuration")

    model_name = st.selectbox("KGE model", list(MAIN_SCRIPTS.keys()), index=0)
    script_name = MAIN_SCRIPTS[model_name]

    try:
        module = load_main_module(script_name)
        defaults = _default_args(module)
    except ModuleLoadError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Inputs")

    dataset_upload = st.file_uploader("Upload dataset archive (.zip)", type=["zip"], key="dataset_upload")
    if dataset_upload is not None:
        dataset_path = _extract_uploaded_archive(dataset_upload, "dataset")
        st.session_state["data_path"] = str(dataset_path)
        st.success("Dataset uploaded successfully.")

    config_upload = st.file_uploader("Upload config.json", type=["json"], key="config_upload")
    if config_upload is not None:
        config_path = _persist_uploaded_file(config_upload, "config.json")
        st.session_state["config_path"] = str(config_path)
        st.success("Configuration uploaded successfully.")

    entity_embedding_upload = st.file_uploader(
        "Upload entity embedding (.npy)", type=["npy"], key="entity_embedding_upload"
    )
    if entity_embedding_upload is not None:
        entity_embedding_path = _persist_uploaded_file(entity_embedding_upload, "entity_embedding.npy")
        st.session_state["entity_embedding_path"] = str(entity_embedding_path)
        st.success("Entity embedding uploaded successfully.")

    relation_embedding_upload = st.file_uploader(
        "Upload relation embedding (.npy)", type=["npy"], key="relation_embedding_upload"
    )
    if relation_embedding_upload is not None:
        relation_embedding_path = _persist_uploaded_file(relation_embedding_upload, "relation_embedding.npy")
        st.session_state["relation_embedding_path"] = str(relation_embedding_path)
        st.success("Relation embedding uploaded successfully.")

    checkpoint_upload = st.file_uploader(
        "Upload trained checkpoint (.zip)", type=["zip"], key="checkpoint_upload"
    )
    if checkpoint_upload is not None:
        checkpoint_path = _extract_uploaded_archive(checkpoint_upload, "checkpoint")
        st.session_state["init_checkpoint"] = str(checkpoint_path)
        st.success("Checkpoint uploaded successfully.")

    data_path = st.session_state.get("data_path", "")
    init_checkpoint = st.session_state.get("init_checkpoint", "")
    config_path = st.session_state.get("config_path", "")
    entity_embedding_path = st.session_state.get("entity_embedding_path", "")
    relation_embedding_path = st.session_state.get("relation_embedding_path", "")
    save_path = _default_output_directory()

    if data_path:
        st.caption(f"Dataset extracted to: `{data_path}`")
    if init_checkpoint:
        st.caption(f"Checkpoint extracted to: `{init_checkpoint}`")
    if config_path:
        st.caption(f"Configuration stored at: `{config_path}`")
    if entity_embedding_path:
        st.caption(f"Entity embedding stored at: `{entity_embedding_path}`")
    if relation_embedding_path:
        st.caption(f"Relation embedding stored at: `{relation_embedding_path}`")
    st.caption(f"Model outputs will be stored in: `{save_path}`")

    do_train = st.checkbox("Train KG model", value=bool(defaults.get("do_train", True)))
    do_valid = st.checkbox("Run validation", value=bool(defaults.get("do_valid", True)))
    do_test = st.checkbox("Run test", value=bool(defaults.get("do_test", True)))
    evaluate_train = st.checkbox("Evaluate train set", value=bool(defaults.get("evaluate_train", False)))

    cuda_available = torch.cuda.is_available()
    cuda_default = bool(defaults.get("cuda", True) and cuda_available)
    use_cuda = st.checkbox("Use CUDA", value=cuda_default, disabled=not cuda_available)
    cuda_device = st.text_input("CUDA device", value=str(defaults.get("cuda_device", "cuda")))
    cuda_visible_devices = st.text_input("CUDA_VISIBLE_DEVICES", value=os.environ.get("CUDA_VISIBLE_DEVICES", "3"))

    with st.expander("Training hyperparameters", expanded=False):
        batch_size = st.number_input("Batch size", min_value=1, value=int(defaults.get("batch_size", 512)))
        negative_sample_size = st.number_input(
            "Negative sample size", min_value=1, value=int(defaults.get("negative_sample_size", 1024))
        )
        hidden_dim = st.number_input("Hidden dimension", min_value=1, value=int(defaults.get("hidden_dim", 500)))
        gamma = st.number_input("Gamma", value=float(defaults.get("gamma", 6.0)))
        adversarial_temperature = st.number_input(
            "Adversarial temperature", value=float(defaults.get("adversarial_temperature", 0.5)), format="%.4f"
        )
        learning_rate = st.number_input(
            "Learning rate", value=float(defaults.get("learning_rate", 5e-5)), format="%.6f"
        )
        max_steps = st.number_input("Max training steps", min_value=1, value=int(defaults.get("max_steps", 80000)))
        save_checkpoint_steps = st.number_input(
            "Checkpoint interval", min_value=1, value=int(defaults.get("save_checkpoint_steps", 10000))
        )
        valid_steps = st.number_input("Validation interval", min_value=1, value=int(defaults.get("valid_steps", 10000)))
        log_steps = st.number_input("Log interval", min_value=1, value=int(defaults.get("log_steps", 100)))
        test_batch_size = st.number_input(
            "Test batch size", min_value=1, value=int(defaults.get("test_batch_size", 8))
        )
        valid_batch_size = st.number_input(
            "Validation batch size", min_value=1, value=int(defaults.get("valid_batch_size", 5))
        )
        cpu_num = st.number_input("CPU workers", min_value=1, value=int(defaults.get("cpu_num", 16)))
        regularization = st.number_input(
            "Regularization (L3)", min_value=0.0, value=float(defaults.get("regularization", 0.0)), format="%.6f"
        )

    with st.expander("KGEC hyperparameters", expanded=False):
        num_bins = st.number_input("Number of bins", min_value=1, value=int(defaults.get("KGEC_num_bins", 10)))
        kgec_lr = st.number_input(
            "KGEC learning rate", min_value=1e-6, value=float(defaults.get("KGEC_learning_rate", 0.01)), format="%.6f"
        )
        initial_temperature = st.number_input(
            "Initial temperature", value=float(defaults.get("KGEC_initial_temperature", 1.0)), format="%.4f"
        )

    run_button = st.button("Run KGEC pipeline")


if run_button:
    if not data_path:
        st.error("Please upload a dataset archive before running the pipeline.")
        st.stop()

    has_checkpoint = bool(init_checkpoint)
    has_config_bundle = bool(config_path and entity_embedding_path and relation_embedding_path)

    if not has_checkpoint and not has_config_bundle and not do_train:
        missing_inputs = []
        if not config_path:
            missing_inputs.append("config.json")
        if not entity_embedding_path:
            missing_inputs.append("entity embedding")
        if not relation_embedding_path:
            missing_inputs.append("relation embedding")

        if missing_inputs:
            missing_desc = ", ".join(missing_inputs)
            st.error(
                "Please upload either a trained checkpoint or provide the following files before running the pipeline: "
                f"{missing_desc}."
            )
        else:
            st.error(
                "Please upload either a trained checkpoint or provide config and embedding files before running the pipeline."
            )
        st.stop()

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices or "3"

    args_updates = {
        "data_path": data_path,
        "save_path": save_path or init_checkpoint,
        "init_checkpoint": init_checkpoint,
        "config_path": config_path or None,
        "entity_embedding_path": entity_embedding_path or None,
        "relation_embedding_path": relation_embedding_path or None,
        "do_train": do_train,
        "do_valid": do_valid,
        "do_test": do_test,
        "evaluate_train": evaluate_train,
        "cuda": bool(use_cuda and torch.cuda.is_available()),
        "cuda_device": cuda_device if use_cuda and torch.cuda.is_available() else "cpu",
        "batch_size": batch_size,
        "negative_sample_size": negative_sample_size,
        "hidden_dim": hidden_dim,
        "gamma": gamma,
        "adversarial_temperature": adversarial_temperature,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "save_checkpoint_steps": save_checkpoint_steps,
        "valid_steps": valid_steps,
        "log_steps": log_steps,
        "test_batch_size": test_batch_size,
        "valid_batch_size": valid_batch_size,
        "cpu_num": cpu_num,
        "regularization": regularization,
        "KGEC_num_bins": num_bins,
        "KGEC_learning_rate": kgec_lr,
        "KGEC_initial_temperature": initial_temperature,
    }

    if not args_updates["save_path"]:
        args_updates["save_path"] = defaults.get("save_path") or defaults.get("init_checkpoint")
    if not args_updates["init_checkpoint"]:
        args_updates["init_checkpoint"] = defaults.get("init_checkpoint")

    args_updates.setdefault("double_entity_embedding", defaults.get("double_entity_embedding", False))
    args_updates.setdefault("double_relation_embedding", defaults.get("double_relation_embedding", False))
    args_updates.setdefault("negative_adversarial_sampling", defaults.get("negative_adversarial_sampling", True))
    args_updates.setdefault("uni_weight", defaults.get("uni_weight", True))
    args_updates.setdefault("test_log_steps", defaults.get("test_log_steps", 1000))

    with st.spinner("Executing KGEC pipeline. This may take a while depending on the configuration..."):
        try:
            result = execute_pipeline(module, args_updates)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            st.error(f"Pipeline execution failed: {exc}")
            st.stop()

    st.success("Pipeline finished successfully.")

    st.subheader("Validation Metrics (final run)")
    st.dataframe(
        _format_metric_table(result.validation_final).style.format({"Value": "{:.6f}"}),
        use_container_width=True,
    )

    st.subheader("Test Metrics (before calibration)")
    st.dataframe(
        _format_metric_table(result.test_original).style.format({"Value": "{:.6f}"}),
        use_container_width=True,
    )

    st.subheader("Test Metrics (after calibration)")
    grouped_metrics = _split_calibration_metrics(result.test_calibrated)
    for calibrator_name, metrics in grouped_metrics.items():
        st.markdown(f"**{calibrator_name}**")
        st.dataframe(
            _format_metric_table(metrics).style.format({"Value": "{:.6f}"}),
            use_container_width=True,
        )

    _display_training_history(result.training_history, "Training History")
    _display_training_history(result.validation_history, "Validation History (during training)")
    _display_logs(result.log_text)

