"""Streamlit demo for KGE Calibrator paper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="KGE Calibrator Demo", layout="wide")


def _load_numpy(upload) -> Optional[np.ndarray]:
    """Safely load a numpy file from an upload."""
    if upload is None:
        return None

    upload.seek(0)
    data = np.load(upload, allow_pickle=False)

    # ``np.load`` with allow_pickle=False returns an ``NpzFile`` when the
    # file contains multiple arrays. We keep the behaviour but fall back to
    # the first array for convenience.
    if isinstance(data, np.lib.npyio.NpzFile):
        first_key = list(data.files)[0]
        return data[first_key]
    return data


def _read_probabilities(upload) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract probabilities and optional labels from a CSV upload.

    The function supports flexible schemas and will infer the most
    appropriate columns when possible. When no column is suitable, a
    synthetic dataset is returned.
    """
    if upload is None:
        return np.empty(0), None

    upload.seek(0)
    df = pd.read_csv(upload)
    if df.empty:
        return np.empty(0), None

    probability_candidates = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
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
        return np.empty(0), None

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


def placeholder_kgec(probabilities: np.ndarray, num_bins: int, lr: float) -> np.ndarray:
    """Placeholder for the KGEC calibration routine.

    This implementation applies a smooth temperature scaling driven by the
    configured hyperparameters. The real KGEC logic can replace this
    function without changing the Streamlit interface.
    """
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(probabilities / (1 - probabilities))
    temperature = 1.0 + (50 - num_bins) / 75.0 + (0.1 - lr) * 2
    calibrated_logits = logits / max(temperature, 1e-3)
    calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
    return calibrated_probs


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


def run_calibration(
    entity_embeddings: Optional[np.ndarray],
    relation_embeddings: Optional[np.ndarray],
    raw_probabilities: np.ndarray,
    raw_labels: Optional[np.ndarray],
    num_bins: int,
    learning_rate: float,
) -> CalibrationResult:
    num_entities = 0 if entity_embeddings is None else entity_embeddings.shape[0]
    num_relations = 0 if relation_embeddings is None else relation_embeddings.shape[0]
    sample_size = max(100, min(1000, num_entities + num_relations))

    if raw_probabilities.size == 0:
        probabilities, labels = _synthesise_predictions(sample_size, seed=num_entities + num_relations)
    else:
        probabilities = np.clip(raw_probabilities, 1e-6, 1 - 1e-6)
        if raw_labels is None:
            rng = np.random.default_rng(num_entities + num_relations or probabilities.size)
            labels = rng.binomial(1, probabilities)
        else:
            labels = raw_labels
            if labels.shape != probabilities.shape:
                labels = np.resize(labels, probabilities.shape)
                labels = np.clip(labels, 0, 1)
    probabilities_before = probabilities
    probabilities_after = placeholder_kgec(probabilities_before, num_bins, learning_rate)

    ece_before = compute_ece(probabilities_before, labels)
    ece_after = compute_ece(probabilities_after, labels)
    ace_before = compute_ace(probabilities_before, labels)
    ace_after = compute_ace(probabilities_after, labels)
    nll_before = compute_nll(probabilities_before, labels)
    nll_after = compute_nll(probabilities_after, labels)

    return CalibrationResult(
        probabilities_before=probabilities_before,
        probabilities_after=probabilities_after,
        labels=labels,
        ece_before=ece_before,
        ece_after=ece_after,
        ace_before=ace_before,
        ace_after=ace_after,
        nll_before=nll_before,
        nll_after=nll_after,
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


def render_ablation_toggle(result: CalibrationResult) -> None:
    st.subheader("Ablation Study (Placeholder)")
    st.info(
        "This section can be extended with ablation metrics or visualisations "
        "once experimental results are available. Replace this message with "
        "domain-specific insights when ready."
    )


def main() -> None:
    st.title("KGE Calibrator Demo")
    st.markdown(
        """
        **KGE Calibrator (KGEC)** enhances the trustworthiness of link prediction models
        by aligning predicted probabilities with observed outcomes. This demo showcases
        how KGEC can post-process Knowledge Graph Embedding (KGE) model outputs to deliver
        better-calibrated confidence scores.

        - 📄 EMNLP 2025 Paper: *KGE Calibrator: An Efficient Probability Calibration Method of Knowledge Graph Embedding Models for Trustworthy Link Prediction*
        - 💻 GitHub: [KGE-Calibrator Repository](https://github.com/your-repo/KGE-Calibrator)
        - 📝 Citation: *Author et al., EMNLP 2025*
        """
    )

    with st.sidebar:
        st.header("Inputs")
        entity_upload = st.file_uploader("Entity embeddings (.npy)", type=["npy"], key="entity")
        relation_upload = st.file_uploader("Relation embeddings (.npy)", type=["npy"], key="relation")
        scores_upload = st.file_uploader("Prediction scores (.csv, optional)", type=["csv"], key="scores")

        st.header("KGEC Hyperparameters")
        num_bins = st.slider("KGEC number of bins", min_value=5, max_value=50, value=10, step=1)
        learning_rate = st.selectbox("Learning rate", options=[0.1, 0.01, 0.001], index=1)

        show_ablation = st.checkbox("Show ablation placeholder", value=False)

        run_demo = st.button("Run Calibration")

    if not run_demo:
        st.info("Upload embeddings, configure the hyperparameters, and click **Run Calibration** to begin.")
        return

    if entity_upload is None or relation_upload is None:
        st.warning("Please upload both entity and relation embedding files to continue.")
        return

    entity_embeddings = _load_numpy(entity_upload)
    relation_embeddings = _load_numpy(relation_upload)
    raw_probabilities, raw_labels = _read_probabilities(scores_upload)

    with st.expander("Embedding overview", expanded=False):
        st.write("Entity embeddings shape:", None if entity_embeddings is None else entity_embeddings.shape)
        st.write("Relation embeddings shape:", None if relation_embeddings is None else relation_embeddings.shape)

    result = run_calibration(
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        raw_probabilities=raw_probabilities,
        raw_labels=raw_labels,
        num_bins=num_bins,
        learning_rate=float(learning_rate),
    )

    render_metrics(result)
    render_histograms(result)

    if show_ablation:
        render_ablation_toggle(result)


if __name__ == "__main__":
    main()
