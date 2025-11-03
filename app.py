"""Streamlit demo for KGE Calibrator paper."""
from __future__ import annotations

import json
from dataclasses import dataclass
import datetime as dt
import io
import os
import pickle
import re
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import entropy

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from torch_uncertainty.metrics import AdaptiveCalibrationError, CalibrationError, CategoricalNLL
except ImportError:  # pragma: no cover
    AdaptiveCalibrationError = CalibrationError = CategoricalNLL = None  # type: ignore

from KGEC_method import KGEC


st.set_page_config(page_title="KGE Calibrator Demo", layout="wide")


def _ascend(path: Path, levels: int) -> Path:
    current = path
    for _ in range(levels):
        current = current.parent
    return current


APP_DIR = Path(__file__).resolve().parent
ROTATE_ROOT = _ascend(APP_DIR, 2)
try:
    PROJECT_ROOT = _ascend(APP_DIR, 4)
except IndexError:
    PROJECT_ROOT = ROTATE_ROOT


DEFAULT_DATASETS: Dict[str, Dict[str, Path]] = {
    "FB15k-237": {
        "entity_embeddings": PROJECT_ROOT / "RAG_Demo" / "models" / "kge_model" / "entity_embedding.npy",
        "relation_embeddings": PROJECT_ROOT / "RAG_Demo" / "models" / "kge_model" / "relation_embedding.npy",
        "config": PROJECT_ROOT / "RAG_Demo" / "models" / "kge_model" / "config.json",
        "triples": ROTATE_ROOT / "data" / "FB15k-237" / "valid.txt",
        "entity_dict": ROTATE_ROOT / "data" / "FB15k-237" / "entities.dict",
        "relation_dict": ROTATE_ROOT / "data" / "FB15k-237" / "relations.dict",
    }
}


def _list_subdirectories(root: Path) -> Dict[str, Path]:
    if not root.exists():
        return {}
    return {path.name: path for path in root.iterdir() if path.is_dir()}


def _discover_datasets() -> Dict[str, Path]:
    candidates: Dict[str, Path] = {}
    data_root = ROTATE_ROOT / "data"
    for name, path in _list_subdirectories(data_root).items():
        if (path / "entities.dict").exists() and (path / "relations.dict").exists():
            candidates[name] = path
    return candidates


def _discover_models() -> Dict[str, Path]:
    models_root = ROTATE_ROOT / "models"
    return {name: path for name, path in _list_subdirectories(models_root).items() if (path / "config.json").exists()}


def _run_main_rotate(
    data_path: Path,
    init_checkpoint: Path,
    save_path: Path,
    num_bins: int,
    learning_rate: float,
    init_temp: float,
    max_steps: int,
    additional_args: Optional[List[str]],
    log_queue: "queue.Queue[str]",
    status_queue: "queue.Queue[str]",
) -> subprocess.Popen:
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"streamlit-{timestamp}.log"

    cmd = [
        sys.executable,
        str(APP_DIR / "Main-RotatE.py"),
        "--data_path",
        str(data_path),
        "--init_checkpoint",
        str(init_checkpoint),
        "--save_path",
        str(save_path),
        "--test_log_name",
        log_name,
        "--KGEC_num_bins",
        str(num_bins),
        "--KGEC_learning_rate",
        str(learning_rate),
        "--KGEC_initial_temperature",
        str(init_temp),
        "--max_steps",
        str(max_steps),
    ]

    if additional_args:
        cmd.extend(additional_args)

    env = os.environ.copy()
    # Allow overriding GPU selection if user toggles to CPU via environment.
    env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    process = subprocess.Popen(
        cmd,
        cwd=str(APP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    stdout_thread = threading.Thread(
        target=_stream_reader,
        args=(process.stdout, log_queue, "STDOUT"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_reader,
        args=(process.stderr, log_queue, "STDERR"),
        daemon=True,
    )
    monitor_thread = threading.Thread(
        target=_process_watchdog,
        args=(process, status_queue),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    monitor_thread.start()

    return process


def _stream_reader(stream: Optional[io.TextIOBase], log_queue: "queue.Queue[str]", label: str) -> None:
    if stream is None:
        return
    for line in iter(stream.readline, ""):
        log_queue.put(f"[{label}] {line.rstrip()}")
    stream.close()


def _process_watchdog(process: subprocess.Popen, status_queue: "queue.Queue[str]") -> None:
    process.wait()
    status_queue.put(f"Process finished with return code {process.returncode}")


@dataclass
class PreparedDataset:
    name: str
    entity_embeddings: np.ndarray
    relation_embeddings: np.ndarray
    scores: np.ndarray
    label_matrix: np.ndarray
    samples: pd.DataFrame
    jump_index: int


@dataclass
class CalibrationResult:
    probabilities_before: np.ndarray
    probabilities_after: np.ndarray
    labels: np.ndarray

    ece_before: float
    ece_after: float
    ace_before: float
    ace_after: float
    nll_before: float
    nll_after: float

    examples: Optional[pd.DataFrame] = None
    dataset_name: Optional[str] = None
    calibrator: Optional[KGEC] = None
    probabilities_matrix_before: Optional[np.ndarray] = None
    probabilities_matrix_after: Optional[np.ndarray] = None
    labels_matrix: Optional[np.ndarray] = None


def _load_numpy(upload) -> Optional[np.ndarray]:
    """Safely load a numpy file from an upload."""
    if upload is None:
        return None

    upload.seek(0)
    file_bytes = upload.read()
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")
    buffer = io.BytesIO(file_bytes)
    buffer.seek(0)
    data = np.load(buffer, allow_pickle=False)

    if isinstance(data, np.lib.npyio.NpzFile):
        first_key = list(data.files)[0]
        return data[first_key]
    return data


def _load_numpy_path(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path, allow_pickle=False)


def _read_probabilities(upload) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract probabilities and optional labels from a CSV upload."""
    if upload is None:
        return np.empty(0, dtype=float), None

    upload.seek(0)
    df = pd.read_csv(upload)
    if df.empty:
        return np.empty(0, dtype=float), None

    probability_candidates = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
    ]

    probability_col = None
    for candidate in probability_candidates:
        lowered = candidate.lower()
        if "prob" in lowered or "score" in lowered:
            probability_col = candidate
            break

    if probability_col is None and probability_candidates:
        probability_col = probability_candidates[0]

    if probability_col is None:
        return np.empty(0, dtype=float), None

    probabilities = df[probability_col].to_numpy(dtype=float)

    label_col = None
    for candidate in df.columns:
        lowered = candidate.lower()
        if "label" in lowered or lowered in {"y", "target"}:
            label_col = candidate
            break

    labels = None
    if label_col is not None:
        labels = df[label_col].to_numpy(dtype=float)

    return probabilities, labels


def _synthesise_predictions(
    n_samples: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(loc=0.0, scale=1.25, size=n_samples)
    probabilities = 1 / (1 + np.exp(-logits))
    labels = rng.binomial(1, probabilities)
    return probabilities, labels


def compute_ece(probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.digitize(probabilities, bins, right=True)
    ece = 0.0

    for bin_index in range(1, len(bins)):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        prob_bin = probabilities[mask]
        label_bin = labels[mask]
        avg_confidence = prob_bin.mean()
        accuracy = label_bin.mean()
        ece += prob_bin.size / probabilities.size * abs(avg_confidence - accuracy)
    return float(ece)


def compute_ace(probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    if probabilities.size == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(probabilities, quantiles)
    # Make sure edges are strictly increasing to avoid empty bins
    bin_edges = np.unique(bin_edges)
    if bin_edges.size <= 1:
        return 0.0

    bin_ids = np.digitize(probabilities, bin_edges, right=True)
    ace = 0.0
    for edge_index in range(1, bin_edges.size):
        mask = bin_ids == edge_index
        if not np.any(mask):
            continue
        prob_bin = probabilities[mask]
        label_bin = labels[mask]
        avg_confidence = prob_bin.mean()
        accuracy = label_bin.mean()
        ace += prob_bin.size / probabilities.size * abs(avg_confidence - accuracy)
    return float(ace)


def compute_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    nll = -(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
    return float(np.mean(nll))


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def _select_jump_index(probabilities: np.ndarray) -> int:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return 0

    sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
    num_candidates = sorted_probs.shape[1]
    kl_values = []
    for i in range(1, num_candidates):
        prev_col = np.clip(sorted_probs[:, i - 1], 1e-12, None)
        current_col = np.clip(sorted_probs[:, i], 1e-12, None)
        kl = entropy(prev_col, current_col)
        kl_values.append(kl)
    if not kl_values:
        return 0
    return int(np.argmax(kl_values))


def _load_id_maps(path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                idx_str, name = line.split("\t")
            except ValueError:
                continue
            idx = int(idx_str)
            id_to_name[idx] = name
            name_to_id[name] = idx
    return id_to_name, name_to_id


def _load_triples(path: Path, entity_index: Dict[str, int], relation_index: Dict[str, int]) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            head, relation, tail = raw_line.strip().split("\t")
            if head not in entity_index or tail not in entity_index or relation not in relation_index:
                continue
            triples.append((entity_index[head], relation_index[relation], entity_index[tail]))
    return triples


def _compute_rotate_scores(
    entity_embeddings: np.ndarray,
    relation_embeddings: np.ndarray,
    head_ids: np.ndarray,
    relation_ids: np.ndarray,
    tail_ids: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    model_name = config.get("model", "RotatE").lower()
    if model_name != "rotate":
        raise ValueError(f"Demo currently supports RotatE embeddings (found model={config.get('model')!r}).")

    gamma = float(config.get("gamma", 9.0))
    hidden_dim = int(config.get("hidden_dim", entity_embeddings.shape[1] // 2))
    epsilon = 2.0
    embedding_range = (gamma + epsilon) / hidden_dim

    if entity_embeddings.shape[1] % 2 != 0:
        raise ValueError("Entity embeddings must have an even dimension for RotatE.")

    heads = entity_embeddings[head_ids]
    relations = relation_embeddings[relation_ids]
    tails = entity_embeddings[tail_ids]

    re_head, im_head = np.split(heads, 2, axis=-1)
    re_tail, im_tail = np.split(tails, 2, axis=-1)

    phase_relation = relations / (embedding_range / np.pi)
    re_relation = np.cos(phase_relation)
    im_relation = np.sin(phase_relation)

    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation
    diff_re = re_score - re_tail
    diff_im = im_score - im_tail

    stacked = np.stack([diff_re, diff_im], axis=0)
    return gamma - np.linalg.norm(stacked, axis=0).sum(axis=-1)


def _prepare_default_dataset(
    dataset_key: str,
    sample_size: int,
    negatives_per_example: int,
    seed: int = 7,
) -> PreparedDataset:
    if dataset_key not in DEFAULT_DATASETS:
        raise KeyError(f"Unknown dataset key: {dataset_key}")

    assets = DEFAULT_DATASETS[dataset_key]
    required_paths = [
        assets["entity_embeddings"],
        assets["relation_embeddings"],
        assets["config"],
        assets["triples"],
        assets["entity_dict"],
        assets["relation_dict"],
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for demo dataset: {', '.join(missing)}")

    entity_embeddings = _load_numpy_path(assets["entity_embeddings"])
    relation_embeddings = _load_numpy_path(assets["relation_embeddings"])
    with assets["config"].open("r", encoding="utf-8") as cfg_handle:
        config = json.load(cfg_handle)
    entity_id_to_name, entity_name_to_id = _load_id_maps(assets["entity_dict"])
    relation_id_to_name, relation_name_to_id = _load_id_maps(assets["relation_dict"])
    triples = _load_triples(assets["triples"], entity_name_to_id, relation_name_to_id)

    if not triples:
        raise RuntimeError("Unable to load triples for the bundled dataset.")

    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, len(triples))
    selected_indices = rng.choice(len(triples), size=sample_size, replace=False)
    selected_triples = np.array([triples[idx] for idx in selected_indices], dtype=int)

    head_ids = selected_triples[:, 0]
    relation_ids = selected_triples[:, 1]
    tail_ids = selected_triples[:, 2]

    positive_scores = _compute_rotate_scores(
        entity_embeddings, relation_embeddings, head_ids, relation_ids, tail_ids, config
    )

    negatives_per_example = max(1, negatives_per_example)
    neg_tail_ids = rng.integers(0, entity_embeddings.shape[0], size=(sample_size, negatives_per_example))
    clash_mask = neg_tail_ids == tail_ids[:, None]
    while clash_mask.any():
        neg_tail_ids[clash_mask] = rng.integers(0, entity_embeddings.shape[0], size=int(clash_mask.sum()))
        clash_mask = neg_tail_ids == tail_ids[:, None]

    neg_head_ids = np.repeat(head_ids, negatives_per_example)
    neg_relation_ids = np.repeat(relation_ids, negatives_per_example)
    neg_tail_flat = neg_tail_ids.reshape(-1)
    negative_scores_flat = _compute_rotate_scores(
        entity_embeddings, relation_embeddings, neg_head_ids, neg_relation_ids, neg_tail_flat, config
    )
    negative_scores = negative_scores_flat.reshape(sample_size, negatives_per_example)

    num_candidates = negatives_per_example + 1
    score_matrix = np.zeros((sample_size, num_candidates), dtype=float)
    label_matrix = np.zeros_like(score_matrix)
    records: List[Dict[str, Any]] = []

    for idx, (h_id, r_id, t_id) in enumerate(selected_triples):
        scores_row = np.concatenate(([positive_scores[idx]], negative_scores[idx]))
        probs_row = _softmax(scores_row)

        score_matrix[idx] = scores_row
        label_row = np.zeros(num_candidates, dtype=float)
        label_row[0] = 1.0
        label_matrix[idx] = label_row

        records.append(
            {
                "example_index": idx,
                "candidate_index": 0,
                "matrix_index": idx * num_candidates,
                "candidate_type": "positive",
                "candidate_rank": 0,
                "label": 1,
                "head_id": int(h_id),
                "relation_id": int(r_id),
                "tail_id": int(t_id),
                "head_name": entity_id_to_name.get(int(h_id), f"Entity {h_id}"),
                "relation_name": relation_id_to_name.get(int(r_id), f"Relation {r_id}"),
                "tail_name": entity_id_to_name.get(int(t_id), f"Entity {t_id}"),
                "score_before": float(scores_row[0]),
                "prob_before": float(probs_row[0]),
            }
        )

        for neg_rank in range(negatives_per_example):
            tail_idx = int(neg_tail_ids[idx, neg_rank])
            candidate_index = neg_rank + 1
            records.append(
                {
                    "example_index": idx,
                    "candidate_index": candidate_index,
                    "matrix_index": idx * num_candidates + candidate_index,
                    "candidate_type": "negative",
                    "candidate_rank": candidate_index,
                    "label": 0,
                    "head_id": int(h_id),
                    "relation_id": int(r_id),
                    "tail_id": tail_idx,
                    "head_name": entity_id_to_name.get(int(h_id), f"Entity {h_id}"),
                    "relation_name": relation_id_to_name.get(int(r_id), f"Relation {r_id}"),
                    "tail_name": entity_id_to_name.get(tail_idx, f"Entity {tail_idx}"),
                    "score_before": float(scores_row[candidate_index]),
                    "prob_before": float(probs_row[candidate_index]),
                }
            )

    samples = pd.DataFrame.from_records(records)

    jump_index = _select_jump_index(_softmax(score_matrix, axis=1))

    return PreparedDataset(
        name=dataset_key,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        scores=score_matrix,
        label_matrix=label_matrix,
        samples=samples,
        jump_index=jump_index,
    )


def _apply_kgec(
    logits: np.ndarray,
    label_matrix: np.ndarray,
    num_bins: int,
    learning_rate: float,
    jump_index: int,
    init_temp: float,
) -> Tuple[np.ndarray, Optional[KGEC]]:
    if logits.size == 0:
        return np.empty_like(logits), None

    if torch is None:
        raise RuntimeError("PyTorch is required to run KGEC calibration.")

    logits_tensor = torch.from_numpy(logits.astype(np.float32))
    labels_tensor = torch.from_numpy(label_matrix.astype(np.float32))

    calibrator = KGEC(num_bins=num_bins, lr=learning_rate, init_temp=init_temp)
    calibrator.fit(logits_tensor, labels_tensor, jump_index=jump_index)

    with torch.no_grad():
        calibrated_tensor = calibrator.predict(logits_tensor)

    calibrated = calibrated_tensor.cpu().numpy()
    calibrated = np.clip(calibrated, 1e-8, None)
    row_sums = np.clip(calibrated.sum(axis=1, keepdims=True), 1e-8, None)
    calibrated = calibrated / row_sums
    calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
    return calibrated, calibrator


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _tensor_to_numpy(data: Any) -> np.ndarray:
    if torch is not None and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)


def _parse_log_metrics(log_text: str) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    pattern = re.compile(
        r"INFO\s+(?P<mode>.+?)\s+(?P<metric>[A-Za-z0-9@_]+)\s+at step\s+(?P<step>-?\d+):\s+(?P<value>-?\d+\.\d+|-?\d+\.?\d*[eE][-+]?\d+|-?\d+\.?\d*)"
    )
    for line in log_text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        mode = match.group("mode").strip()
        metric = match.group("metric").strip()
        value = float(match.group("value"))
        metrics.setdefault(mode, {})[metric] = value
    return metrics


def _calibration_metrics_with_uncertainty(
    matrix_before: np.ndarray,
    matrix_after: np.ndarray,
    label_matrix: np.ndarray,
) -> Tuple[float, float, float, float, float, float]:
    assert matrix_before.shape == matrix_after.shape == label_matrix.shape

    if (
        torch is not None
        and CalibrationError is not None
        and AdaptiveCalibrationError is not None
        and CategoricalNLL is not None
    ):
        targets = np.argmax(label_matrix, axis=1).astype(np.int64)
        probs_before_tensor = torch.from_numpy(matrix_before.astype(np.float32))
        probs_after_tensor = torch.from_numpy(matrix_after.astype(np.float32))
        targets_tensor = torch.from_numpy(targets)

        ece_before_metric = CalibrationError(
            task="multiclass", num_classes=matrix_before.shape[1]
        )
        ece_before_metric.update(probs_before_tensor, targets_tensor)
        ece_before = float(ece_before_metric.compute().item())

        ece_after_metric = CalibrationError(
            task="multiclass", num_classes=matrix_after.shape[1]
        )
        ece_after_metric.update(probs_after_tensor, targets_tensor)
        ece_after = float(ece_after_metric.compute().item())

        ace_before_metric = AdaptiveCalibrationError(
            task="multiclass", num_classes=matrix_before.shape[1]
        )
        ace_before_metric.update(probs_before_tensor, targets_tensor)
        ace_before = float(ace_before_metric.compute().item())

        ace_after_metric = AdaptiveCalibrationError(
            task="multiclass", num_classes=matrix_after.shape[1]
        )
        ace_after_metric.update(probs_after_tensor, targets_tensor)
        ace_after = float(ace_after_metric.compute().item())

        nll_before_metric = CategoricalNLL()
        nll_before_metric.update(probs_before_tensor, targets_tensor)
        nll_before = float(nll_before_metric.compute().item())

        nll_after_metric = CategoricalNLL()
        nll_after_metric.update(probs_after_tensor, targets_tensor)
        nll_after = float(nll_after_metric.compute().item())

        return ece_before, ece_after, ace_before, ace_after, nll_before, nll_after

    # Fallback: use numpy-based calculations on the positive-class column.
    single_before = np.clip(matrix_before[:, 0], 1e-6, 1 - 1e-6)
    single_after = np.clip(matrix_after[:, 0], 1e-6, 1 - 1e-6)
    labels = label_matrix[:, 0]
    ece_before = compute_ece(single_before, labels)
    ece_after = compute_ece(single_after, labels)
    ace_before = compute_ace(single_before, labels)
    ace_after = compute_ace(single_after, labels)
    nll_before = compute_nll(single_before, labels)
    nll_after = compute_nll(single_after, labels)
    return ece_before, ece_after, ace_before, ace_after, nll_before, nll_after


def run_calibration(
    probabilities: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    num_bins: int,
    learning_rate: float,
    init_temp: float,
    examples: Optional[pd.DataFrame] = None,
    dataset_name: Optional[str] = None,
    logits_matrix: Optional[np.ndarray] = None,
    label_matrix: Optional[np.ndarray] = None,
    jump_index: int = -1,
) -> CalibrationResult:
    using_multiclass = logits_matrix is not None

    if using_multiclass:
        logits_matrix = np.asarray(logits_matrix, dtype=float)
        if label_matrix is None:
            raise RuntimeError("Label matrix is required when supplying logits.")
        label_matrix = np.asarray(label_matrix, dtype=float)
        probabilities_matrix_before = _softmax(logits_matrix, axis=1)
        probabilities_before = probabilities_matrix_before.reshape(-1)
        labels_vector = label_matrix.reshape(-1)
        logits_for_calibrator = logits_matrix
        labels_for_calibrator = label_matrix
    else:
        if probabilities is None or np.size(probabilities) == 0:
            probabilities, labels_generated = _synthesise_predictions(512)
            labels = labels_generated
            dataset_name = dataset_name or "Synthetic (demo)"
            examples = None
        else:
            probabilities = np.asarray(probabilities, dtype=float)
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        if labels is None:
            rng = np.random.default_rng(probabilities.size)
            labels = rng.binomial(1, probabilities)
        else:
            labels = np.asarray(labels, dtype=float)
            if labels.shape != probabilities.shape:
                labels = np.resize(labels, probabilities.shape)
            labels = np.clip(labels, 0.0, 1.0)

        labels_vector = labels
        probabilities_before = probabilities
        logits_for_calibrator = np.log(np.stack([probabilities_before, 1.0 - probabilities_before], axis=1))
        labels_for_calibrator = np.stack([labels_vector, 1.0 - labels_vector], axis=1)
        probabilities_matrix_before = np.stack([probabilities_before, 1.0 - probabilities_before], axis=1)

    probabilities_before = np.clip(probabilities_before, 1e-6, 1 - 1e-6)

    if using_multiclass and jump_index < 0:
        jump_index = _select_jump_index(probabilities_matrix_before)
    elif not using_multiclass and jump_index < 0:
        jump_index = 0

    calibrated_matrix, calibrator = _apply_kgec(
        logits_for_calibrator,
        labels_for_calibrator,
        num_bins=num_bins,
        learning_rate=learning_rate,
        jump_index=jump_index,
        init_temp=init_temp,
    )

    if using_multiclass:
        probabilities_matrix_after = calibrated_matrix
        probabilities_after = probabilities_matrix_after.reshape(-1)
        labels_eval = labels_vector
    else:
        probabilities_matrix_after = calibrated_matrix
        probabilities_after = np.clip(probabilities_matrix_after[:, 0], 1e-6, 1 - 1e-6)
        labels_eval = labels_vector
        probabilities_before = probabilities_matrix_before[:, 0]

    probabilities_after = np.clip(probabilities_after, 1e-6, 1 - 1e-6)

    (
        ece_before,
        ece_after,
        ace_before,
        ace_after,
        nll_before,
        nll_after,
    ) = _calibration_metrics_with_uncertainty(
        probabilities_matrix_before,
        probabilities_matrix_after,
        labels_for_calibrator,
    )

    examples_with_predictions = None
    if examples is not None:
        flat_before = probabilities_matrix_before.reshape(-1)
        flat_after = probabilities_matrix_after.reshape(-1)
        examples_with_predictions = examples.copy()
        examples_with_predictions["prob_after"] = np.nan
        if "matrix_index" in examples_with_predictions.columns:
            matrix_indices = examples_with_predictions["matrix_index"].to_numpy(dtype=int)
            valid_mask = (matrix_indices >= 0) & (matrix_indices < flat_after.size)
            if np.any(valid_mask):
                examples_with_predictions.loc[valid_mask, "prob_after"] = flat_after[matrix_indices[valid_mask]]
                examples_with_predictions.loc[valid_mask, "prob_before"] = flat_before[matrix_indices[valid_mask]]
        elif len(examples_with_predictions) == flat_after.size:
            examples_with_predictions["prob_after"] = flat_after
        else:
            fill_count = min(len(examples_with_predictions), flat_after.size)
            examples_with_predictions.loc[: fill_count - 1, "prob_after"] = flat_after[:fill_count]
        if "prob_before" in examples_with_predictions.columns:
            examples_with_predictions["prob_after"] = examples_with_predictions["prob_after"].fillna(
                examples_with_predictions["prob_before"]
            )

    return CalibrationResult(
        probabilities_before=probabilities_before,
        probabilities_after=probabilities_after,
        labels=labels_eval,
        ece_before=ece_before,
        ece_after=ece_after,
        ace_before=ace_before,
        ace_after=ace_after,
        nll_before=nll_before,
        nll_after=nll_after,
        examples=examples_with_predictions,
        dataset_name=dataset_name,
        calibrator=calibrator,
        probabilities_matrix_before=probabilities_matrix_before,
        probabilities_matrix_after=probabilities_matrix_after,
        labels_matrix=labels_for_calibrator,
    )


def render_metrics(result: CalibrationResult) -> None:
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Expected Calibration Error", "Adaptive Calibration Error", "Negative Log-Likelihood"],
            "Before KGEC": [result.ece_before, result.ace_before, result.nll_before],
            "After KGEC": [result.ece_after, result.ace_after, result.nll_after],
        }
    )
    st.subheader("Calibration Metrics")
    st.dataframe(metrics_df.style.format({"Before KGEC": "{:.4f}", "After KGEC": "{:.4f}"}), use_container_width=True)

    long_df = metrics_df.melt(id_vars="Metric", var_name="Stage", value_name="Value")
    chart = (
        alt.Chart(long_df)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X("Metric", sort=None),
            y=alt.Y("Value", title="Score"),
            color=alt.Color("Stage", scale=alt.Scale(scheme="tableau20")),
            column=alt.Column("Stage", header=alt.Header(title=""), spacing=10),
            tooltip=["Stage", "Metric", alt.Tooltip("Value", format=".4f")],
        )
        .properties(width=200, height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def render_histograms(result: CalibrationResult) -> None:
    st.subheader("Confidence Histograms")
    before_df = pd.DataFrame({"Probability": result.probabilities_before, "Stage": "Before KGEC"})
    after_df = pd.DataFrame({"Probability": result.probabilities_after, "Stage": "After KGEC"})
    hist_df = pd.concat([before_df, after_df], ignore_index=True)

    hist_chart = (
        alt.Chart(hist_df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("Probability", bin=alt.Bin(maxbins=25), title="Predicted Probability"),
            y=alt.Y("count()", title="Count"),
            color=alt.Color("Stage", scale=alt.Scale(scheme="set2")),
            facet=alt.Facet("Stage", columns=2, header=alt.Header(labelOrient="bottom")),
        )
        .properties(height=300)
    )
    st.altair_chart(hist_chart, use_container_width=True)


def render_example_analysis(result: CalibrationResult, show_section: bool) -> None:
    if not show_section or result.examples is None or result.examples.empty:
        return

    positives = result.examples[result.examples["label"] == 1].copy()
    if positives.empty:
        st.info("No labelled positives available for example exploration.")
        return

    positives = positives.sort_values(by="prob_before", ascending=False).reset_index(drop=True)
    positives["display"] = positives.apply(
        lambda row: f"{row['head_name']} — {row['relation_name']} → {row['tail_name']}", axis=1
    )

    st.subheader("Example Explorer")
    selection = st.selectbox("Choose a positive triple", positives["display"])
    selected_row = positives.loc[positives["display"] == selection].iloc[0]
    selected_idx = int(selected_row["example_index"])
    subset = result.examples[result.examples["example_index"] == selected_idx].copy()

    subset["Candidate"] = np.where(
        subset["label"] == 1,
        "Ground truth",
        subset["tail_name"],
    )
    subset = subset.sort_values(by="prob_after", ascending=False).reset_index(drop=True)
    display_df = subset[
        [
            "Candidate",
            "candidate_type",
            "prob_before",
            "prob_after",
            "tail_name",
            "label",
        ]
    ].rename(
        columns={
            "candidate_type": "Type",
            "prob_before": "Probability (before)",
            "prob_after": "Probability (after)",
            "tail_name": "Tail entity",
            "label": "Label",
        }
    )
    display_df["Label"] = display_df["Label"].map({1: "Positive", 0: "Negative"})

    st.markdown(
        f"**Query:** `{selected_row['head_name']}` — `{selected_row['relation_name']}`"
    )
    st.dataframe(display_df.drop(columns=["Tail entity"]), use_container_width=True)

    chart_df = subset.assign(
        Label=subset["label"].map({1: "Positive", 0: "Negative"}),
    ).melt(
        id_vars=["tail_name", "Label"],
        value_vars=["prob_before", "prob_after"],
        var_name="Stage",
        value_name="Probability",
    )
    chart = (
        alt.Chart(chart_df)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X("tail_name", title="Tail candidate"),
            y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Stage", scale=alt.Scale(scheme="tableau20")),
            column=alt.Column("Label", header=alt.Header(title=""), spacing=8),
            tooltip=[
                "Stage",
                alt.Tooltip("tail_name", title="Tail"),
                alt.Tooltip("Probability", format=".4f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def main() -> None:
    st.title("KGE Calibrator Demo")
    st.markdown(
        """
        **KGE Calibrator (KGEC)** enhances the trustworthiness of link prediction models
        by aligning predicted probabilities with observed outcomes. This Streamlit app now
        wraps the original `Main-RotatE.py` pipeline so you can launch a full calibration
        run, inspect every log, and visualise the before/after behaviour without leaving
        your browser.
        """
    )

    dataset_options = _discover_datasets()
    model_options = _discover_models()

    with st.sidebar:
        st.header("Main-RotatE Configuration")
        dataset_name = st.selectbox(
            "Knowledge graph dataset",
            options=sorted(dataset_options.keys()),
            help="Datasets discovered under `Calibration/RotatE-master/data`.",
        )
        model_name = st.selectbox(
            "Pretrained checkpoint",
            options=sorted(model_options.keys()),
            help="Checkpoints discovered under `Calibration/RotatE-master/models`.",
        )
        num_bins = st.slider("KGEC number of bins", min_value=5, max_value=50, value=10, step=1)
        learning_rate = float(
            st.selectbox("KGEC learning rate", options=[0.1, 0.01, 0.001], format_func=lambda v: f"{v:g}", index=1)
        )
        init_temp = st.number_input("KGEC initial temperature", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        max_steps = st.number_input(
            "Training steps (set 0 to reuse checkpoint)",
            min_value=0,
            max_value=100000,
            value=0,
            step=1000,
            help="Passes through the `--max_steps` flag; 0 skips additional training.",
        )
        extra_cli = st.text_input(
            "Extra CLI arguments",
            value="",
            help="Optional additional flags passed directly to Main-RotatE.py (e.g. `--log_steps 1000`).",
        )
        show_logs = st.checkbox("Show log contents", value=True)
        run_button = st.button("Run Main-RotatE")

    if not dataset_options or not model_options:
        st.error(
            "Unable to locate datasets or checkpoints. Please ensure "
            "`Calibration/RotatE-master/data` and `Calibration/RotatE-master/models` are populated."
        )
        return

    if not run_button:
        st.info("Configure the run on the left and click **Run Main-RotatE** to start calibration.")
        return

    dataset_path = dataset_options[dataset_name]
    model_path = model_options[model_name]

    run_root = APP_DIR / "streamlit_runs"
    run_dir = run_root / f"{dataset_name}-{model_name}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    additional_args: List[str] = []
    if extra_cli.strip():
        additional_args = shlex.split(extra_cli)

    log_queue_obj: "queue.Queue[str]" = queue.Queue()
    status_queue_obj: "queue.Queue[str]" = queue.Queue()
    process = _run_main_rotate(
        data_path=dataset_path,
        init_checkpoint=model_path,
        save_path=run_dir,
        num_bins=num_bins,
        learning_rate=learning_rate,
        init_temp=init_temp,
        max_steps=max_steps,
        additional_args=additional_args,
        log_queue=log_queue_obj,
        status_queue=status_queue_obj,
    )

    progress_placeholder = st.info("Running Main-RotatE.py... check live logs below.")
    status_placeholder = st.empty()
    live_log_placeholder = st.empty() if show_logs else None

    log_lines: List[str] = []
    status_messages: List[str] = []

    def _drain_queue(q: "queue.Queue[str]") -> List[str]:
        drained: List[str] = []
        while True:
            try:
                drained.append(q.get_nowait())
            except queue.Empty:
                break
        return drained

    while True:
        new_lines = _drain_queue(log_queue_obj)
        if new_lines:
            log_lines.extend(new_lines)
            if show_logs and live_log_placeholder is not None:
                preview = "\n".join(log_lines[-400:])
                live_log_placeholder.code(preview or "(no output yet)", language="text")

        new_status = _drain_queue(status_queue_obj)
        if new_status:
            status_messages.extend(new_status)
            status_placeholder.write("\n".join(status_messages))
            if any("return code" in msg for msg in new_status):
                break

        if process.poll() is not None and not new_lines and not new_status:
            break

        time.sleep(0.1)

    # Flush any remaining output after process termination.
    log_lines.extend(_drain_queue(log_queue_obj))
    status_messages.extend(_drain_queue(status_queue_obj))
    if status_messages:
        status_placeholder.write("\n".join(status_messages))

    process.wait()
    returncode = process.returncode or 0

    if live_log_placeholder is not None and log_lines:
        live_log_placeholder.code("\n".join(log_lines[-400:]), language="text")

    if returncode == 0:
        progress_placeholder.success("Main-RotatE.py completed successfully.")
    else:
        progress_placeholder.error(f"Main-RotatE.py exited with return code {returncode}.")

    st.subheader("Execution Summary")
    st.write(f"Run directory: `{run_dir}`")
    st.write(f"Return code: {returncode}")

    full_log_text = "\n".join(log_lines)
    with st.expander("Captured log (STDOUT/STDERR)", expanded=False):
        st.code(full_log_text or "(no output captured)", language="text")

    if returncode != 0:
        st.error("Main-RotatE.py exited with a non-zero status. Check the captured log for details.")
        return

    log_path = None
    for candidate in ("train.log", "test.log"):
        candidate_path = run_dir / candidate
        if candidate_path.exists():
            log_path = candidate_path
            break

    log_text = ""
    if log_path and show_logs:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        with st.expander(f"Log file: {log_path.name}", expanded=False):
            st.code(log_text, language="text")

    parsed_log_metrics: Dict[str, Dict[str, float]] = {}
    if log_text:
        parsed_log_metrics = _parse_log_metrics(log_text)
        if parsed_log_metrics:
            st.subheader("Logged Metrics")
            for mode, values in parsed_log_metrics.items():
                clean_mode = mode.rstrip(":")
                df = pd.DataFrame([values], index=[clean_mode]).T.rename(columns={clean_mode: "Value"})
                st.write(f"**{clean_mode}**")
                st.dataframe(df.style.format("{:.6f}"), use_container_width=True)

    metrics_file = run_dir / "metrics_dict_valid"
    scores_file = run_dir / "all_model_score_valid"
    labels_file = run_dir / "all_positive_arg_valid"

    if metrics_file.exists():
        metrics_dict = _load_pickle(metrics_file)
        if isinstance(metrics_dict, dict):
            aggregated = {
                metric: float(np.mean(values)) if values else float("nan")
                for metric, values in metrics_dict.items()
            }
            st.subheader("Validation Metrics (from pickle)")
            st.dataframe(pd.DataFrame([aggregated]).T.rename(columns={0: "Value"}).style.format("{:.6f}"))

    if not scores_file.exists() or not labels_file.exists():
        st.warning(
            "Calibration tensors were not generated (expected `all_model_score_valid` and `all_positive_arg_valid`). "
            "Nothing to visualise."
        )
        return

    try:
        raw_scores = _tensor_to_numpy(_load_pickle(scores_file))
        label_matrix = _tensor_to_numpy(_load_pickle(labels_file)).astype(float)
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to load calibration tensors: {exc}")
        return

    dataset_label = f"{dataset_name} (validation head/tail batches)"
    probabilities_before_matrix = _softmax(raw_scores, axis=1)
    derived_jump = _select_jump_index(probabilities_before_matrix)

    try:
        calibration_result = run_calibration(
            probabilities=None,
            labels=None,
            num_bins=num_bins,
            learning_rate=learning_rate,
            init_temp=init_temp,
            examples=None,
            dataset_name=dataset_label,
            logits_matrix=raw_scores,
            label_matrix=label_matrix,
            jump_index=derived_jump,
        )
    except RuntimeError as err:
        st.error(f"Post-processing with KGEC failed: {err}")
        return

    st.caption(
        f"Calibrated on {dataset_label} ({label_matrix.shape[0]} queries, "
        f"{label_matrix.shape[1]} candidates each; jump index = {derived_jump})."
    )

    render_metrics(calibration_result)
    render_histograms(calibration_result)
    st.info(
        "Example-level exploration requires entity and relation labels, which are not "
        "persisted by Main-RotatE. The visualisations above reflect aggregate behaviour."
    )


if __name__ == "__main__":
    main()
