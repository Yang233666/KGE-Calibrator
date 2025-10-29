import altair as alt
import numpy as np
import pandas as pd
import torch
import streamlit as st

from dataclasses import dataclass
from typing import Dict, List, Optional

from calibration_training import calibrate_and_evaluate_from_scores
from KGEC_method import KGEC, UncalCalibrator


st.set_page_config(page_title="KGE Calibrator Demo", layout="wide")


@dataclass
class CalibrationResult:
    original_metrics: Dict[str, float]
    calibrate_metrics: Dict[str, float]
    calibrator_names: List[str]
    top1_before: np.ndarray
    top1_after: Dict[str, np.ndarray]


def _load_array(upload) -> Optional[np.ndarray]:
    """Load a NumPy array from an uploaded file."""

    if upload is None:
        return None

    upload.seek(0)
    array = np.load(upload, allow_pickle=False)
    if isinstance(array, np.lib.npyio.NpzFile):
        first_key = list(array.files)[0]
        return array[first_key]
    return array


def _validate_inputs(
    valid_scores: np.ndarray,
    valid_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
) -> None:
    """Validate array shapes to ensure compatibility with the calibration pipeline."""

    if valid_scores.ndim != 2:
        raise ValueError("Validation scores must be a 2-D array of shape [n_samples, n_entities].")
    if test_scores.ndim != 2:
        raise ValueError("Test scores must be a 2-D array of shape [n_samples, n_entities].")

    if valid_scores.shape[1] != test_scores.shape[1]:
        raise ValueError("Validation and test scores must have the same number of entities (columns).")

    if valid_labels.ndim not in (1, 2):
        raise ValueError("Validation labels must be 1-D indices or 2-D one-hot encodings.")
    if test_labels.ndim not in (1, 2):
        raise ValueError("Test labels must be 1-D indices or 2-D one-hot encodings.")

    if valid_labels.shape[0] != valid_scores.shape[0]:
        raise ValueError("Validation labels must align with the validation score rows.")
    if test_labels.shape[0] != test_scores.shape[0]:
        raise ValueError("Test labels must align with the test score rows.")


def _run_calibration(
    valid_scores: np.ndarray,
    valid_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    num_bins: int,
    learning_rate: float,
    initial_temperature: float,
) -> CalibrationResult:
    calibration_models = [
        UncalCalibrator(),
        KGEC(num_bins=num_bins, init_temp=initial_temperature, lr=learning_rate),
    ]

    original_metrics, calibrate_metrics = calibrate_and_evaluate_from_scores(
        valid_scores=valid_scores,
        valid_labels=valid_labels,
        test_scores=test_scores,
        test_labels=test_labels,
        calibration_models_list=calibration_models,
        nentity=test_scores.shape[1],
    )

    test_scores_tensor = torch.as_tensor(test_scores, dtype=torch.float32)
    top1_before = test_scores_tensor.softmax(dim=1).cpu().numpy().max(axis=1)

    top1_after: Dict[str, np.ndarray] = {}
    for calibration_model in calibration_models:
        calibrated_probs = calibration_model.predict(test_scores_tensor).cpu().numpy()
        top1_after[calibration_model.name] = calibrated_probs.max(axis=1)

    return CalibrationResult(
        original_metrics=original_metrics,
        calibrate_metrics=calibrate_metrics,
        calibrator_names=[model.name for model in calibration_models],
        top1_before=top1_before,
        top1_after=top1_after,
    )


def render_metrics(result: CalibrationResult) -> None:
    ranking_metrics = ["MRR", "MR", "HITS@1", "HITS@3", "HITS@10"]
    calibration_metrics = ["ECE", "ACE", "NLL"]

    ranking_rows = []
    for metric in ranking_metrics:
        row = {"Metric": metric, "Original": result.original_metrics.get(metric, np.nan)}
        for name in result.calibrator_names:
            row[name] = result.calibrate_metrics.get(f"{name}_{metric}", np.nan)
        ranking_rows.append(row)

    ranking_df = pd.DataFrame(ranking_rows)
    st.subheader("Ranking Metrics")
    st.dataframe(
        ranking_df.style.format(lambda value: f"{value:.4f}" if isinstance(value, (float, np.floating)) else value),
        use_container_width=True,
    )

    calibration_rows = []
    for metric in calibration_metrics:
        row = {"Metric": metric}
        for name in result.calibrator_names:
            row[name] = result.calibrate_metrics.get(f"{name}_{metric}", np.nan)
        calibration_rows.append(row)

    calibration_df = pd.DataFrame(calibration_rows)
    st.subheader("Calibration Metrics")
    st.dataframe(
        calibration_df.style.format(lambda value: f"{value:.4f}" if isinstance(value, (float, np.floating)) else value),
        use_container_width=True,
    )

    long_df = ranking_df.melt(id_vars="Metric", var_name="Stage", value_name="Value")
    chart = (
        alt.Chart(long_df)
        .mark_bar(opacity=0.85)
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
    st.subheader("Confidence Histograms (Top-1 Probabilities)")

    records = [{"Probability": value, "Stage": "Original"} for value in result.top1_before]
    for name, values in result.top1_after.items():
        records.extend({"Probability": value, "Stage": name} for value in values)

    hist_df = pd.DataFrame.from_records(records)

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


def main() -> None:
    st.title("KGE Calibrator Demo")
    st.markdown(
        """
        This demo reproduces the Knowledge Graph Embedding Calibrator (KGEC)
        results using the exact scoring tensors generated by the original
        training scripts. Upload the validation and test tensors produced by
        the script run to verify that the calibration metrics match the
        published numbers.
        """
    )

    with st.sidebar:
        st.header("Inputs")
        valid_scores_upload = st.file_uploader(
            "Validation model scores (.npy or .npz)", type=["npy", "npz"], key="valid_scores"
        )
        valid_labels_upload = st.file_uploader(
            "Validation positive labels (.npy or .npz)", type=["npy", "npz"], key="valid_labels"
        )
        test_scores_upload = st.file_uploader(
            "Test model scores (.npy or .npz)", type=["npy", "npz"], key="test_scores"
        )
        test_labels_upload = st.file_uploader(
            "Test positive labels (.npy or .npz)", type=["npy", "npz"], key="test_labels"
        )

        st.header("KGEC Hyperparameters")
        num_bins = st.slider("KGEC number of bins", min_value=5, max_value=50, value=10, step=1)
        learning_rate = st.selectbox("Learning rate", options=[0.1, 0.01, 0.001], index=1)
        initial_temperature = st.slider("Initial temperature", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

        show_hist = st.checkbox("Show confidence histograms", value=True)
        run_demo = st.button("Run Calibration")

    if not run_demo:
        st.info("Upload the tensors generated by the original script and click **Run Calibration** to begin.")
        return

    valid_scores = _load_array(valid_scores_upload)
    valid_labels = _load_array(valid_labels_upload)
    test_scores = _load_array(test_scores_upload)
    test_labels = _load_array(test_labels_upload)

    if None in (valid_scores, valid_labels, test_scores, test_labels):
        st.warning("Please upload all four tensors before running the calibration.")
        return

    try:
        _validate_inputs(valid_scores, valid_labels, test_scores, test_labels)
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.expander("Tensor overview", expanded=False):
        st.write("Validation scores shape:", valid_scores.shape)
        st.write("Validation labels shape:", valid_labels.shape)
        st.write("Test scores shape:", test_scores.shape)
        st.write("Test labels shape:", test_labels.shape)

    result = _run_calibration(
        valid_scores=valid_scores,
        valid_labels=valid_labels,
        test_scores=test_scores,
        test_labels=test_labels,
        num_bins=num_bins,
        learning_rate=float(learning_rate),
        initial_temperature=float(initial_temperature),
    )

    render_metrics(result)

    if show_hist:
        render_histograms(result)


if __name__ == "__main__":
    main()
