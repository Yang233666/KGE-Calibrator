#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KGE Calibrator — WWW Demo (v0.2, supports big checkpoints)
==========================================================

This version solves the 200MB upload limit in three ways:
1) **URL fetch**: paste a direct download URL to the checkpoint/dataset ZIP. The server downloads it.
2) **Embeddings path**: upload `entity_embedding.npy` + `relation_embedding.npy` + `config.json`
   instead of a >300MB checkpoint, then the app builds KGEModel and assigns weights directly.
3) **Multi-part uploads**: upload `checkpoint.part01`, `checkpoint.part02`, ... (each < 200MB) and
   we reassemble into the original ZIP.

All computation still reuses your original code:
- `KGEModel.test_step(..., calibrate=True)` for valid+training KGEC
- `KGEModel.calibration_predict(...)` for test metrics
- `KGEC` from `KGEC_method.py`

Run locally:
    $ pip install -r requirements.txt
    $ pip install streamlit requests
    $ streamlit run app.py

To raise Streamlit's upload limit locally, add a `.streamlit/config.toml` in the same folder:
    [server]
    maxUploadSize = 2048   # MB

Author: prepared for WWW demo track
"""
import io
import os
import gc
import re
import json
import time
import shutil
import zipfile
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import streamlit as st
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Optional: network fetch (for URL mode)
try:
    import requests
except Exception:
    requests = None

# ---- Reuse your original code (core computation) ----
from calibration_training import KGEModel
from KGEC_method import *  # KGEC class
from dataloader import TestDataset

# --- Small utils (IO) ---
def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def find_file(root: str, names: List[str]) -> Optional[str]:
    for dirpath, _, filenames in os.walk(root):
        for nm in names:
            if nm in filenames:
                return os.path.join(dirpath, nm)
    return None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def extract_zip_to_dir(zip_bytes: bytes, out_dir: str) -> str:
    zip_fp = os.path.join(out_dir, "bundle.zip")
    with open(zip_fp, "wb") as f:
        f.write(zip_bytes)
    with zipfile.ZipFile(zip_fp, "r") as z:
        z.extractall(out_dir)
    # If single top folder, return it
    entries = [os.path.join(out_dir, e) for e in os.listdir(out_dir) if e != "bundle.zip"]
    if len(entries) == 1 and os.path.isdir(entries[0]):
        return entries[0]
    return out_dir

def reassemble_parts(files) -> bytes:
    """Concatenate multiple uploaded file-like objects in lexicographic order of their names."""
    parts = sorted(files, key=lambda f: getattr(f, "name", ""))
    buf = io.BytesIO()
    for part in parts:
        buf.write(part.read())
    return buf.getvalue()

def download_url(url: str) -> bytes:
    if requests is None:
        raise RuntimeError("`requests` not installed. Run: pip install requests")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        return r.content

def find_required_dataset_files(data_root: str) -> Tuple[str, str, str, str, str]:
    ents = find_file(data_root, ["entities.dict"])
    rels = find_file(data_root, ["relations.dict"])
    train = find_file(data_root, ["train.txt"])
    valid = find_file(data_root, ["valid.txt"])
    test = find_file(data_root, ["test.txt"])
    missing = [n for n, p in {"entities.dict": ents, "relations.dict": rels,
                              "train.txt": train, "valid.txt": valid, "test.txt": test}.items() if p is None]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")
    return ents, rels, train, valid, test

def load_id_maps(entities_path: str, relations_path: str):
    entity2id, relation2id = {}, {}
    with open(entities_path) as fin:
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
    with open(relations_path) as fin:
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    return entity2id, relation2id

def load_config_from_dir(checkpoint_or_cfg_dir: str) -> Dict:
    cfg_path = os.path.join(checkpoint_or_cfg_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError("config.json not found.")
    with open(cfg_path, "r") as f:
        return json.load(f)

# --- Args plumbing ---
@dataclass
class DemoArgs:
    data_path: str = ""
    save_path: str = ""
    init_checkpoint: str = ""
    model: str = "RotatE"
    double_entity_embedding: bool = True
    double_relation_embedding: bool = False
    hidden_dim: int = 500
    gamma: float = 6.0
    nentity: int = 0
    nrelation: int = 0
    test_batch_size: int = 8
    cpu_num: int = 8
    cuda: bool = False
    cuda_device: str = "cuda:0"
    test_log_steps: int = 1000
    KGEC_num_bins: int = 10
    KGEC_learning_rate: float = 0.01
    KGEC_initial_temperature: float = 1.0

def override_args_from_config(args: DemoArgs, cfg: Dict) -> DemoArgs:
    args.model = cfg.get("model", args.model)
    args.double_entity_embedding = cfg.get("double_entity_embedding", args.double_entity_embedding)
    args.double_relation_embedding = cfg.get("double_relation_embedding", args.double_relation_embedding)
    args.hidden_dim = cfg.get("hidden_dim", args.hidden_dim)
    args.gamma = cfg.get("gamma", args.gamma)
    args.test_batch_size = cfg.get("test_batch_size", args.test_batch_size)
    args.data_path = cfg.get("data_path", args.data_path)
    args.nentity = cfg.get("nentity", args.nentity)
    args.nrelation = cfg.get("nrelation", args.nrelation)
    return args

def make_namespace(args: DemoArgs):
    return SimpleNamespace(**vars(args))

def build_calibration_models(args: DemoArgs):
    models = []
    kgec = KGEC(
        num_bins=args.KGEC_num_bins,
        lr=args.KGEC_learning_rate,
        init_temp=args.KGEC_initial_temperature
    )
    models.append(kgec)
    return models

def set_global_determinism(seed: int = 2025):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# --- Embedding loader (for .npy path) ---
def apply_numpy_embeddings_to_model(model: KGEModel, entity_npy: bytes, relation_npy: bytes, device: torch.device):
    ent = np.load(io.BytesIO(entity_npy))
    rel = np.load(io.BytesIO(relation_npy))
    # Convert to float32 tensors
    ent_t = torch.from_numpy(ent).to(device=device, dtype=torch.float32)
    rel_t = torch.from_numpy(rel).to(device=device, dtype=torch.float32)

    if model.entity_embedding.shape != ent_t.shape:
        raise ValueError(f"Entity embedding shape mismatch: model {tuple(model.entity_embedding.shape)} vs npy {tuple(ent_t.shape)}")
    if model.relation_embedding.shape != rel_t.shape:
        raise ValueError(f"Relation embedding shape mismatch: model {tuple(model.relation_embedding.shape)} vs npy {tuple(rel_t.shape)}")

    with torch.no_grad():
        model.entity_embedding.data.copy_(ent_t)
        model.relation_embedding.data.copy_(rel_t)

# --------------------------- Streamlit UI ---------------------------
st.set_page_config(page_title="KGE Calibrator — WWW Demo (v0.2)", layout="wide")

st.title("KGE Calibrator — WWW Demo (v0.2)")
st.caption("Three ways to bring your model in: **URL fetch**, **multi‑part upload**, or **.npy embeddings**.")

with st.expander("🔧 Demo configuration", expanded=True):
    colA, colB, colC, colD = st.columns(4)
    with colA:
        use_cuda = st.toggle("Use CUDA (if available)", value=torch.cuda.is_available())
    with colB:
        cuda_device = st.text_input("CUDA device", value="cuda:0")
    with colC:
        test_bs = st.number_input("Test batch size", min_value=1, max_value=4096, value=8, step=1)
    with colD:
        cpu_num = st.number_input("CPU worker threads", min_value=1, max_value=os.cpu_count() or 8, value=8, step=1)

    st.subheader("KGEC hyperparameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_bins = st.number_input("KGEC num_bins", min_value=2, max_value=200, value=10, step=1)
    with col2:
        lr = st.number_input("KGEC learning_rate", min_value=1e-6, max_value=1.0, value=0.01, step=0.01, format="%.6f")
    with col3:
        init_temp = st.number_input("KGEC initial_temperature", min_value=0.01, max_value=100.0, value=1.0, step=0.1)

st.markdown("---")

# Input modes
tab_zip, tab_url, tab_npy = st.tabs(["📦 Upload ZIP / Multi‑part", "🌐 Download from URL", "🧪 Embeddings (.npy)"])

# Shared dataset inputs (for all modes)
with st.sidebar:
    st.header("Dataset (required for all modes)")
    ds_upload = st.file_uploader("Dataset ZIP (entities.dict, relations.dict, train/valid/test.txt)", type=["zip"], key="ds_zip")
    st.caption("Alternatively paste a URL if the dataset is large:")
    ds_url = st.text_input("Dataset ZIP URL (optional)", key="ds_url")

def prepare_dataset(ds_upload, ds_url) -> str:
    workdir = tempfile.mkdtemp(prefix="kgec_data_")
    if ds_upload is not None:
        byts = ds_upload.read()
    elif ds_url:
        byts = download_url(ds_url)
    else:
        st.error("Please upload a dataset ZIP or provide a dataset URL.")
        st.stop()
    data_dir = extract_zip_to_dir(byts, workdir)
    # Validate
    find_required_dataset_files(data_dir)
    return data_dir

# --- Mode 1: ZIP / Multi-part
with tab_zip:
    st.subheader("Checkpoint ZIP upload (single)")
    ckpt_zip = st.file_uploader("Checkpoint ZIP (contains `config.json` + `checkpoint`)", type=["zip"], key="ckpt_zip")

    st.subheader("OR: Multi‑part uploads")
    st.caption("Split your ZIP into parts below 200MB (e.g., `split -b 190m checkpoint.zip checkpoint.part`) and upload all parts.")
    ckpt_parts = st.file_uploader("Checkpoint parts", type=None, key="ckpt_parts", accept_multiple_files=True)

    run1 = st.button("🚀 Run with uploaded ZIP / parts", key="run_zip")

# --- Mode 2: URL fetch
with tab_url:
    st.subheader("Checkpoint URL")
    ckpt_url = st.text_input("Direct download URL to checkpoint ZIP (must contain `config.json` + `checkpoint`)", key="ckpt_url")
    run2 = st.button("🚀 Run with URL", key="run_url")

# --- Mode 3: .npy embeddings
with tab_npy:
    st.subheader(".npy Embeddings + config.json")
    ent_npy = st.file_uploader("entity_embedding.npy", type=["npy"], key="ent_npy")
    rel_npy = st.file_uploader("relation_embedding.npy", type=["npy"], key="rel_npy")
    cfg_npy = st.file_uploader("config.json (from the original training)", type=["json"], key="cfg_json")
    run3 = st.button("🚀 Run with .npy embeddings", key="run_npy")

# ------------------- Main run logic -------------------
def run_pipeline(model_builder, cfg_dict, dataset_dir):
    ents, rels, train_fp, valid_fp, test_fp = find_required_dataset_files(dataset_dir)
    entity2id, relation2id = load_id_maps(ents, rels)

    train_triples = read_triple(train_fp, entity2id, relation2id)
    valid_triples = read_triple(valid_fp, entity2id, relation2id)
    test_triples  = read_triple(test_fp,  entity2id, relation2id)
    all_true_triples = train_triples + valid_triples + test_triples

    args = DemoArgs(
        data_path=dataset_dir,
        save_path=tempfile.mkdtemp(prefix="kgec_cache_"),
        test_batch_size=test_bs,
        cpu_num=int(cpu_num),
        cuda=bool(use_cuda and torch.cuda.is_available()),
        cuda_device=cuda_device,
        KGEC_num_bins=int(num_bins),
        KGEC_learning_rate=float(lr),
        KGEC_initial_temperature=float(init_temp),
    )
    args = override_args_from_config(args, cfg_dict)
    ns = make_namespace(args)

    set_global_determinism(seed=2025)

    # Build model (deferred so we can either load checkpoint or apply .npy weights)
    kge_model = model_builder(ns)
    if ns.cuda:
        kge_model = kge_model.to(torch.device(ns.cuda_device))

    # Build calibration models
    calibration_models_list = build_calibration_models(args=ns)

    # Validation (fit KGEC) then test
    with st.spinner("Running validation (training KGEC on valid set)…"):
        valid_metrics = KGEModel.test_step(kge_model, valid_triples, all_true_triples, ns,
                                           calibration_models_list, calibrate=True)
    with st.spinner("Running test prediction…"):
        original_metrics, calibrated_metrics = KGEModel.calibration_predict(
            kge_model, test_triples, all_true_triples, ns, calibration_models_list
        )

    return valid_metrics, original_metrics, calibrated_metrics

def show_results(valid_metrics, original_metrics, calibrated_metrics, note: str):
    st.success("Done! Results below.")
    st.markdown("### ✅ Validation metrics (pre-calibration)")
    st.json(valid_metrics)
    st.markdown("### 📈 Test metrics (original, before calibration)")
    st.json(original_metrics)
    st.markdown("### 🎯 Test metrics (after calibration with **KGEC**)")
    st.json(calibrated_metrics)

    blob = {
        "valid_metrics": valid_metrics,
        "test_original_metrics": original_metrics,
        "test_calibrated_metrics": calibrated_metrics,
        "notes": note
    }
    out_path = os.path.join(tempfile.mkdtemp(prefix="kgec_out_"), "kgec_demo_results.json")
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2)
    with open(out_path, "rb") as f:
        st.download_button("💾 Download results JSON", f.read(), file_name="kgec_demo_results.json", mime="application/json")

# Mode 1: ZIP / parts
if 'run1' in locals() and run1:
    data_dir = prepare_dataset(ds_upload, ds_url)
    workdir = tempfile.mkdtemp(prefix="kgec_ckpt_")

    if ckpt_zip is not None:
        ckpt_dir = extract_zip_to_dir(ckpt_zip.read(), workdir)
    elif ckpt_parts:
        joined = reassemble_parts(ckpt_parts)
        ckpt_dir = extract_zip_to_dir(joined, workdir)
    else:
        st.error("Please upload a checkpoint ZIP or parts.")
        st.stop()

    cfg = load_config_from_dir(ckpt_dir)

    def builder(ns):
        model = KGEModel(ns.model, ns.nentity, ns.nrelation, ns.hidden_dim, ns.gamma,
                         double_entity_embedding=ns.double_entity_embedding,
                         double_relation_embedding=ns.double_relation_embedding)
        # Load checkpoint weights
        ckpt_path = find_file(ckpt_dir, ["checkpoint"])
        if ckpt_path is None:
            raise FileNotFoundError("`checkpoint` not found in the ZIP.")
        state = torch.load(ckpt_path, map_location=("cpu" if not ns.cuda else ns.cuda_device))
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=True)
        else:
            model.load_state_dict(state, strict=False)
        return model

    v, o, c = run_pipeline(builder, cfg, data_dir)
    show_results(v, o, c, note="ZIP / parts mode.")

# Mode 2: URL
if 'run2' in locals() and run2:
    data_dir = prepare_dataset(ds_upload, ds_url)
    if not ckpt_url:
        st.error("Please provide a checkpoint ZIP URL.")
        st.stop()
    workdir = tempfile.mkdtemp(prefix="kgec_ckpt_url_")
    byts = download_url(ckpt_url)
    ckpt_dir = extract_zip_to_dir(byts, workdir)
    cfg = load_config_from_dir(ckpt_dir)

    def builder(ns):
        model = KGEModel(ns.model, ns.nentity, ns.nrelation, ns.hidden_dim, ns.gamma,
                         double_entity_embedding=ns.double_entity_embedding,
                         double_relation_embedding=ns.double_relation_embedding)
        ckpt_path = find_file(ckpt_dir, ["checkpoint"])
        if ckpt_path is None:
            raise FileNotFoundError("`checkpoint` not found in the ZIP from URL.")
        state = torch.load(ckpt_path, map_location=("cpu" if not ns.cuda else ns.cuda_device))
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=True)
        else:
            model.load_state_dict(state, strict=False)
        return model

    v, o, c = run_pipeline(builder, cfg, data_dir)
    show_results(v, o, c, note="URL mode.")

# Mode 3: .npy
if 'run3' in locals() and run3:
    data_dir = prepare_dataset(ds_upload, ds_url)
    if ent_npy is None or rel_npy is None or cfg_npy is None:
        st.error("Please upload entity_embedding.npy, relation_embedding.npy, and config.json.")
        st.stop()

    cfg = json.load(cfg_npy)

    def builder(ns):
        model = KGEModel(ns.model, ns.nentity, ns.nrelation, ns.hidden_dim, ns.gamma,
                         double_entity_embedding=ns.double_entity_embedding,
                         double_relation_embedding=ns.double_relation_embedding)
        device = torch.device(ns.cuda_device if ns.cuda else "cpu")
        apply_numpy_embeddings_to_model(model, ent_npy.read(), rel_npy.read(), device=device)
        return model

    v, o, c = run_pipeline(builder, cfg, data_dir)
    show_results(v, o, c, note=".npy embeddings mode.")
