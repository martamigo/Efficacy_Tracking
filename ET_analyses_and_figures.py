#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code by Marta Migó, MA - formatted using Codex
"""

from __future__ import annotations

import math
import random
import shutil
import warnings
from dataclasses import dataclass
from functools import reduce
from itertools import combinations, product
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import skew, yeojohnson
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import OLSInfluence

try:
    from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
    from sklearn.preprocessing import StandardScaler
    HAS_FACTOR_ANALYZER = True
except Exception:
    HAS_FACTOR_ANALYZER = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "data_May_DC8.csv"
QUESTIONNAIRE_PATH = DATA_DIR / "ET_Screening_August15.csv"
DEMOGRAPHICS_PATH = DATA_DIR / "demographics.csv"

OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_FIT_STARTS = 5
N_SIM_STARTS = 2
N_PARALLEL = 1000
ALPHA = 0.05
P_CORR_METHOD = "fdr_bh"
EXCLUDE_OTHER_GROUP = True
REQUIRE_COMPLETE_CORE_COVARIATES = True
CORE_ANALYSIS_COLUMNS = ["Age", "Sex", "RRS_Score", "DARS_Score", "avg_condition_order"]
MANUSCRIPT_DOCX_PATH = PROJECT_ROOT / "Migo_ET_final_v3.docx"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# =============================================================================
# STYLE / LABELS
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 1.0,
    "savefig.dpi": 600,
})

PURPLE = "#7E57C2"
LIGHT_PURPLE = "#C7B6E6"
GROUP_ORDER = ["LRLA", "HRHA", "HRLA"]
# Manuscript figure order/colors. The analytic GROUP_ORDER above is left unchanged
# for existing models, but the manuscript figures use LRLA, HRLA, HRHA left-to-right.
FIGURE_GROUP_ORDER = ["LRLA", "HRLA", "HRHA"]
GROUP_COLORS = {
    "LRLA": "#8FB8D4",   # soft blue
    "HRLA": "#D9A066",   # soft orange
    "HRHA": "#8FC29B",   # soft green
}
DARS_COLOR = "#8E79C6"
RRS_COLOR = "#7FC7B1"
MODEL_BLUE = "#1f77b4"
MODEL_ORANGE = "#ff7f0e"
RAW_GROUP_TO_PLOT = {
    "low RRS": "LRLA",
    "high RRS, high Anh": "HRHA",
    "high RRS, low Anh": "HRLA",
    "Other": "Other",
}

FINALQ_LABELS = {
    "FinalQ_1": "How aware are you of how your actions impact others?",
    "FinalQ_2": "Are you able to adjust your behavior after receiving feedback?",
    "FinalQ_3": "Do you unfairly blame yourself for things that happen to you?",
    "FinalQ_4": "Do you unfairly blame others/external circumstances for things that happen to you?",
}

Y_LABELS = {
    "sse_total": "SSE, full task (lower = better)",
    "sse_low": "SSE, low-efficacy condition (lower = better)",
    "sse_high": "SSE, high-efficacy condition (lower = better)",
    "log_sse_total": "log-transformed SSE (lower = better)",
    "log_sse_low": "log-transformed SSE (lower = better)",
    "log_sse_high": "log-transformed SSE (lower = better)",
}

TTITLES = {
    "sse_total": "Full-task efficacy-tracking error",
    "sse_low": "Low-efficacy condition error",
    "sse_high": "High-efficacy condition error",
    "log_sse_total": "Full-task efficacy-tracking error",
    "log_sse_low": "Low-efficacy condition error",
    "log_sse_high": "High-efficacy condition error",
}

PRETTY_OUTCOME = {
    "Alpha_p": "Positive learning rate",
    "Alpha_n": "Negative learning rate",
    "alpha_diff": "Positive − negative learning-rate difference",
    "sse_total": "Full-task efficacy-tracking error",
    "sse_low": "Low-efficacy condition error",
    "sse_high": "High-efficacy condition error",
    "sse_diff": "High − low efficacy-tracking error",
    "avg_eff_rating_high": "Average efficacy estimate, high condition",
    "avg_eff_rating_low": "Average efficacy estimate, low condition",
    "avg_diff_eff_minus_true": "Signed efficacy discrepancy, full task",
    "avg_diff_eff_minus_true_high": "Signed efficacy discrepancy, high condition",
    "avg_diff_eff_minus_true_low": "Signed efficacy discrepancy, low condition",
    "Eff_intercept": "Initial efficacy estimate",
}

# =============================================================================
# UTILITIES
# =============================================================================

def beautify_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.9)
    ax.grid(False)


def p_to_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return ""


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def collapse_one_row(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    """Collapse to one row per ID, taking first non-missing value in each column."""
    def first_nonmissing(s):
        s = s.dropna()
        return s.iloc[0] if len(s) else np.nan

    return df.groupby(id_col, as_index=False).agg(first_nonmissing)


def collapse_questionnaires(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    """Keep the best questionnaire row per participant.

    Qualtrics exports can contain repeated rows for the same participant. For
    publication analyses, a later completed row is preferable to an earlier
    partial/duplicate row. This avoids mixing item values across rows and fixes
    participants whose group assignment differs across duplicate submissions.
    """
    q = df.copy()
    q[id_col] = q[id_col].astype(str).str.strip()
    q = q[(q[id_col].notna()) & (q[id_col] != "") & (q[id_col].str.lower() != "nan")].copy()

    sort_cols = [id_col]
    ascending = [True]
    for col in ["Finished", "Progress"]:
        if col in q.columns:
            q[f"__sort_{col}"] = pd.to_numeric(q[col], errors="coerce")
            sort_cols.append(f"__sort_{col}")
            ascending.append(True)
    if "EndDate" in q.columns:
        q["__sort_EndDate"] = pd.to_datetime(q["EndDate"], errors="coerce")
        sort_cols.append("__sort_EndDate")
        ascending.append(True)

    q = q.sort_values(sort_cols, ascending=ascending)
    q = q.drop_duplicates(subset=id_col, keep="last")
    q = q.drop(columns=[c for c in q.columns if c.startswith("__sort_")])
    return q.reset_index(drop=True)


def safe_z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def residualize(data: pd.DataFrame, y: str, covariates: list[str]) -> pd.Series:
    covariates = [c for c in covariates if c in data.columns]
    if not covariates:
        return data[y]
    formula = f"{y} ~ " + " + ".join(covariates)
    return smf.ols(formula, data=data.dropna(subset=[y] + covariates)).fit().resid


def add_sig_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c="black")
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom", fontsize=10)


def assert_one_row_per_participant(df: pd.DataFrame, name: str, id_col: str = "ID") -> None:
    """Fail early if a participant-level table accidentally has repeated IDs."""
    if id_col not in df.columns:
        raise KeyError(f"{name} is missing required ID column: {id_col}")
    n_rows = len(df)
    n_ids = df[id_col].astype(str).nunique(dropna=True)
    if n_rows != n_ids:
        dup_counts = df[id_col].astype(str).value_counts()
        dup_counts = dup_counts[dup_counts > 1].head(15)
        raise ValueError(
            f"{name} must be one row per participant, but has {n_rows} rows "
            f"for {n_ids} unique IDs. Example duplicate IDs/counts:\n{dup_counts}"
        )
    print(f"OK: {name} is participant-level ({n_rows} rows, {n_ids} unique IDs).", flush=True)


def report_dataframe_level(df: pd.DataFrame, name: str, id_col: str = "ID") -> None:
    """Print whether a table is participant-level or trial/row-level."""
    if id_col not in df.columns:
        print(f"{name}: missing {id_col}; columns = {df.columns.tolist()}", flush=True)
        return
    n_rows = len(df)
    n_ids = df[id_col].astype(str).nunique(dropna=True)
    ratio = n_rows / n_ids if n_ids else float("nan")
    level = "participant-level" if n_rows == n_ids else "row/trial-level"
    print(f"{name}: {n_rows} rows, {n_ids} unique IDs, {ratio:.2f} rows/ID → {level}", flush=True)

# =============================================================================
# DATA LOADING AND TASK CLEANING
# =============================================================================

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path)
    # Fixed: use pd.isna rather than `x in [np.nan]`, which never behaves reliably.
    block = raw["Block"].copy()
    for i in range(1, len(block) - 1):
        if pd.isna(block.iloc[i]) and block.iloc[i - 1] == "PR" and block.iloc[i + 1] == "PR":
            block.iloc[i] = "PR"
    raw["Block"] = block
    raw["ID"] = raw["ID"].astype(str).str.strip()
    return raw


def compute_performance_training_noise(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    performance_data = raw_data.loc[
        raw_data["Block"].eq("PR"),
        ["ID", "Point_amount", "Estimated_points", "Correct", "trial_type"],
    ].copy()
    performance_data = performance_data[performance_data["trial_type"].isin(["Performance_tracking", "Feedback"])]

    values = performance_data["Point_amount"].to_numpy()
    estimates = performance_data["Estimated_points"].dropna().to_numpy()
    ids = performance_data["ID"].to_numpy()

    collected_ids, collected_points = [], []
    for i in range(len(values) - 1):
        if pd.isna(values[i]) and not pd.isna(values[i + 1]):
            collected_ids.append(ids[i + 1])
            collected_points.append(values[i + 1])

    n = min(len(collected_ids), len(collected_points), len(estimates))
    result_df = pd.DataFrame({
        "ID": collected_ids[:n],
        "point_amount": collected_points[:n],
        "Estimated_points": estimates[:n],
    })
    result_df["abs_diff"] = (result_df["point_amount"] - result_df["Estimated_points"]).abs()
    result_df["diff"] = result_df["point_amount"] - result_df["Estimated_points"]
    result_df["trial_number"] = result_df.groupby("ID").cumcount() + 1

    per_id_mean_abs_diff = result_df.groupby("ID")["abs_diff"].mean().rename("mean_abs_diff")
    print("\n=== Performance-tracking noise, calibration/training ===")
    print(f"Group mean: {per_id_mean_abs_diff.mean():.6f}")
    print(f"Group SD:   {per_id_mean_abs_diff.std(ddof=1):.6f}")
    print(f"N IDs:      {per_id_mean_abs_diff.shape[0]}")
    return result_df, per_id_mean_abs_diff


def compute_phase2_accuracy(raw_data: pd.DataFrame) -> pd.DataFrame:
    phase2_data = raw_data.loc[
        raw_data["Block"].eq("Block 0"),
        ["ID", "response", "trial_type_feed", "trial_type"],
    ].copy()
    phase2_data = phase2_data[phase2_data["trial_type"].eq("randomornot")].copy()
    phase2_data["Correct_Phase2"] = (
        ((phase2_data["response"].eq("PERFORMANCE-BASED")) & (phase2_data["trial_type_feed"].eq("performance"))) |
        ((phase2_data["response"].eq("RANDOM")) & (phase2_data["trial_type_feed"].eq("random")))
    ).astype(int)
    phase2_data["trial_number"] = phase2_data.groupby("ID").cumcount() + 1

    per_id = phase2_data.groupby("ID")["Correct_Phase2"].agg(n_trials="count", n_correct="sum", acc="mean").reset_index()
    t_res = stats.ttest_1samp(per_id["acc"], popmean=0.5, alternative="greater")

    print("\n=== Phase 2 efficacy-identification accuracy ===")
    print(f"Mean acc = {per_id['acc'].mean()*100:.2f}% (SD = {per_id['acc'].std(ddof=1)*100:.2f}%)")
    print(f"t({len(per_id)-1}) = {t_res.statistic:.3f}, p = {t_res.pvalue:.4g}")
    return phase2_data


def build_clean_task_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data[raw_data["Block"].eq("1")].copy()
    data["ID"] = data["ID"].astype(str).str.strip()
    data["trial"] = data.groupby("ID").cumcount() + 1

    all_rows = []
    for pid, part_data in data.sort_values(["ID", "trial"]).groupby("ID"):
        part_data = part_data.reset_index(drop=True)
        perf_data = part_data[part_data["trial_type"].eq("Performance_tracking")].reset_index(drop=True)
        rt_data = part_data[part_data["trial_type"].isin(["image-keyboard-response", "Green"])]["rt"].dropna().reset_index(drop=True)
        feedback_data = part_data[part_data["trial_type"].eq("Feedback")].reset_index(drop=True)
        red_data = part_data[part_data["trial_type"].eq("Red")]["True_Perc"].dropna().reset_index(drop=True)

        point_amount = part_data["Point_amount"].dropna().reset_index(drop=True)
        estimated_points = perf_data["response"].dropna().astype(float).reset_index(drop=True)
        trial_type_feed = feedback_data["trial_type_feed"].dropna().reset_index(drop=True)
        block = feedback_data["Block"].dropna().reset_index(drop=True)

        trial_types = part_data["trial_type"].reset_index(drop=True)
        responses = part_data["response"].reset_index(drop=True)
        feedback_idx = trial_types[trial_types.eq("Feedback")].index
        next_types = trial_types.shift(-1).loc[feedback_idx]
        next_responses = responses.shift(-1).loc[feedback_idx]
        eff_rating = np.where(next_types.eq("Eff_Rating"), next_responses, np.nan).astype(float) / 100

        rt_min_list = part_data["Speed_min"].dropna().drop_duplicates().tolist() if "Speed_min" in part_data else []
        rt_sd_list = part_data["Speed_sd"].dropna().drop_duplicates().tolist() if "Speed_sd" in part_data else []
        rt_threshold_list = part_data["Speed"].dropna().drop_duplicates().tolist() if "Speed" in part_data else []
        rn_min = rn_sd = 0

        for i in range(len(perf_data)):
            row = {
                "ID": pid,
                "trial": perf_data.loc[i, "trial"] if "trial" in perf_data.columns else np.nan,
                "point_amount": point_amount[i] if i < len(point_amount) else np.nan,
                "estimated_points": estimated_points[i] if i < len(estimated_points) else np.nan,
                "rt": rt_data[i] if i < len(rt_data) else np.nan,
                "trial_type": trial_type_feed[i] if i < len(trial_type_feed) else np.nan,
                "block": block[i] if i < len(block) else np.nan,
                "True_Perc": red_data[i] if i < len(red_data) else np.nan,
                "Eff_Rating": eff_rating[i] if i < len(eff_rating) else np.nan,
                "rt_threshold": rt_threshold_list[0] if rt_threshold_list else np.nan,
                "rt_min": rt_min_list[rn_min] if rt_min_list else np.nan,
                "rt_sd": rt_sd_list[rn_sd] if rt_sd_list else np.nan,
            }
            if i > 0 and "Speed_min" in perf_data.columns and "Speed_sd" in perf_data.columns:
                curr_min = perf_data.loc[i, "Speed_min"]
                curr_sd = perf_data.loc[i, "Speed_sd"]
                if not pd.isna(curr_min) and rt_min_list and curr_min != row["rt_min"]:
                    rn_min = min(rn_min + 1, len(rt_min_list) - 1)
                    row["rt_min"] = rt_min_list[rn_min]
                if not pd.isna(curr_sd) and rt_sd_list and curr_sd != row["rt_sd"]:
                    rn_sd = min(rn_sd + 1, len(rt_sd_list) - 1)
                    row["rt_sd"] = rt_sd_list[rn_sd]
            all_rows.append(row)

    clean_data = pd.DataFrame(all_rows).dropna(subset=["block"]).sort_values(["ID", "trial"]).reset_index(drop=True)

    # Trim IDs with >96 rows to first 96 rows, preserving the correction from the prior notebook.
    row_counts = clean_data.groupby("ID").size()
    ids_over_96 = row_counts[row_counts > 96].index
    clean_data = clean_data.groupby("ID", group_keys=False).apply(
        lambda g: g.head(96) if g.name in ids_over_96 else g
    ).reset_index(drop=True)

    # Recompute performance-based feedback from RT whenever feedback was random.
    perf_feedback = []
    for _, r in clean_data.iterrows():
        if r["trial_type"] == "performance":
            perf_feedback.append(r["point_amount"])
        elif pd.notna(r["rt"]) and pd.notna(r["rt_threshold"]) and pd.notna(r["rt_min"]) and pd.notna(r["rt_sd"]):
            if r["rt"] <= r["rt_threshold"]:
                denom = r["rt_threshold"] - r["rt_min"]
                val = np.round(((r["rt_threshold"] - r["rt"]) * 6) / denom) if denom != 0 else np.nan
            else:
                denom = r["rt_threshold"] + 3 * r["rt_sd"]
                val = np.round(((r["rt"] - r["rt_threshold"]) * -6) / denom) if denom != 0 else np.nan
                val = max(val, -6) if pd.notna(val) else np.nan
            perf_feedback.append(val)
        else:
            perf_feedback.append(np.nan)
    clean_data["Performance_feedback"] = perf_feedback

    clean_data["trial_PE"] = (clean_data["estimated_points"] - clean_data["point_amount"]).abs()
    clean_data["trialtype"] = clean_data["Eff_Rating"].notna().astype(int)
    return clean_data

# =============================================================================
# MODEL
# =============================================================================

def newdist_binary(current_val: float, desired_mean: float) -> int:
    return int(float(current_val) >= float(desired_mean))


def twoLR_Cutoff_intercept_b_drifting_trial(x0, feedback_actual, feedback_performance, accum, trialtype, n_samples):
    alpha_p, alpha_n, ev_cut, intercept = x0
    Ef = []
    Ef_good = []

    for i in range(n_samples):
        first_obs = 1 - (abs(float(feedback_performance.iloc[i]) - float(feedback_actual.iloc[i])) / 12)
        new_obs = newdist_binary(first_obs, ev_cut)

        if len(Ef) == 0:
            Ef.append(float(intercept))

        pe = float(new_obs - Ef[-1])
        lr = alpha_p if pe > 0 else alpha_n
        Ef.append(Ef[-1] + float(lr) * pe)

        if int(trialtype[i]) == 1:
            Ef_good.append(Ef[-1])

    Ef_good = np.asarray(Ef_good, dtype=float)
    accum = np.asarray(accum, dtype=float)
    n = min(len(Ef_good), len(accum))
    if n == 0:
        return np.inf
    return float(np.sum((Ef_good[:n] - accum[:n]) ** 2))


def simulate_data_twoLR_int_drifting_trial(alpha_p, alpha_n, n_samples, evidence_cutoff, first_eff, trialtype, feedback_actual, feedback_performance):
    Ef = []
    accum = []

    for i in range(n_samples):
        first_obs = 1 - (abs(float(feedback_performance.iloc[i]) - float(feedback_actual.iloc[i])) / 12)
        new_obs = newdist_binary(first_obs, evidence_cutoff)

        if len(Ef) == 0:
            Ef.append(float(first_eff))

        pe = float(new_obs - Ef[-1])
        lr = alpha_p if pe > 0 else alpha_n
        Ef.append(Ef[-1] + float(lr) * pe)

        if int(trialtype[i]) == 1:
            accum.append(Ef[-1])

    return np.asarray(accum, dtype=float)


def fit_model_all_participants(clean_data: pd.DataFrame, n_starts: int = N_FIT_STARTS) -> pd.DataFrame:
    rows = []
    clean_ids = clean_data["ID"].drop_duplicates().to_numpy()

    for counter, pid in enumerate(clean_ids, start=1):
        d = clean_data[clean_data["ID"].eq(pid)].reset_index(drop=True)
        feedback_actual = pd.to_numeric(d["point_amount"], errors="coerce")
        feedback_perf = pd.to_numeric(d["estimated_points"], errors="coerce")
        accum = d["Eff_Rating"].dropna().astype(float).to_numpy()
        trialtype = np.where(d["Eff_Rating"].isna(), 0, 1)
        n_samples = len(trialtype)

        rec = []
        for _ in range(n_starts):
            x0 = [random.uniform(0.01, 1), random.uniform(0.01, 1), random.uniform(0.5, 1), random.uniform(0, 1)]
            bnds = ((0.0001, 1), (0.0001, 1), (0.5, 1), (0, 1))
            result = minimize(
                twoLR_Cutoff_intercept_b_drifting_trial,
                x0,
                args=(feedback_actual, feedback_perf, accum, trialtype, n_samples),
                bounds=bnds,
                method="Nelder-Mead",
            )
            rec.append({"sse": result.fun, "Alpha_p": result.x[0], "Alpha_n": result.x[1], "Ev_Cut": result.x[2], "Eff_intercept": result.x[3]})

        best = min(rec, key=lambda x: x["sse"])
        sse = max(best["sse"], np.finfo(float).eps)
        k = 4
        aic = n_samples * np.log(sse / n_samples) + 2 * k
        bic = n_samples * np.log(sse / n_samples) + k * np.log(n_samples)
        rows.append({
            "ID": pid,
            "Model": "twoLR_Cutoff_intercept_b_drifting_trial",
            "Alpha_p": best["Alpha_p"],
            "Alpha_n": best["Alpha_n"],
            "Ev_Cut": best["Ev_Cut"],
            "Eff_intercept": best["Eff_intercept"],
            "AIC": aic,
            "BIC": bic,
            "SSE": sse,
        })
        if counter % 50 == 0 or counter == len(clean_ids):
            print(f"Fitted {counter}/{len(clean_ids)} participants")

    return pd.DataFrame(rows)


def compute_behavioral_metrics(clean_data: pd.DataFrame, gooddata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    for pid in gooddata["ID"].astype(str):
        d = clean_data[clean_data["ID"].astype(str).eq(pid)].reset_index(drop=True)
        valid = d[["True_Perc", "Eff_Rating"]].dropna().index
        y_true = pd.to_numeric(d.loc[valid, "True_Perc"], errors="coerce").to_numpy()
        y_pred = pd.to_numeric(d.loc[valid, "Eff_Rating"], errors="coerce").to_numpy()
        if len(y_true) == 0:
            continue

        high = y_true > 0.5
        low = y_true < 0.5
        diff = y_pred - y_true
        metric_rows.append({
            "ID": pid,
            "sse_total": np.sum((y_true - y_pred) ** 2),
            "sse_high": np.sum((y_true[high] - y_pred[high]) ** 2) if high.any() else np.nan,
            "sse_low": np.sum((y_true[low] - y_pred[low]) ** 2) if low.any() else np.nan,
            "avg_eff_rating_high": np.nanmean(y_pred[high]) if high.any() else np.nan,
            "avg_eff_rating_low": np.nanmean(y_pred[low]) if low.any() else np.nan,
            "avg_diff_eff_minus_true": np.nanmean(diff),
            "avg_diff_eff_minus_true_high": np.nanmean(diff[high]) if high.any() else np.nan,
            "avg_diff_eff_minus_true_low": np.nanmean(diff[low]) if low.any() else np.nan,
            "n_rating_trials": len(y_true),
        })

    metrics = pd.DataFrame(metric_rows)
    metrics["sse_diff"] = metrics["sse_high"] - metrics["sse_low"]

    # Condition order: 1 = high-to-low, 2 = low-to-high.
    order_rows = []
    for pid, g in clean_data.groupby("ID"):
        true_vals = pd.to_numeric(g["True_Perc"], errors="coerce").dropna().to_numpy()
        if len(true_vals) < 2:
            order = np.nan
        else:
            first = true_vals[: len(true_vals) // 2].mean()
            second = true_vals[len(true_vals) // 2 :].mean()
            order = 1 if first > second else 2 if second > first else np.nan
        order_rows.append({"ID": str(pid), "avg_condition_order": order})
    order_df = pd.DataFrame(order_rows)

    metrics = metrics.merge(order_df, on="ID", how="left")
    good = gooddata.copy()
    good["ID"] = good["ID"].astype(str)
    metrics = metrics.merge(good, on="ID", how="left")
    metrics["alpha_diff"] = metrics["Alpha_p"] - metrics["Alpha_n"]
    return metrics, order_df


def build_fit_df(clean_data: pd.DataFrame, gooddata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clean = clean_data.copy()
    clean["ID"] = clean["ID"].astype(str)
    good = gooddata.copy()
    good["ID"] = good["ID"].astype(str)

    for pid in sorted(set(clean["ID"]).intersection(good["ID"])):
        d = clean[clean["ID"].eq(pid)].reset_index(drop=True)
        p = good[good["ID"].eq(pid)].iloc[0]
        trialtype = np.where(d["Eff_Rating"].isna(), 0, 1)
        pred = simulate_data_twoLR_int_drifting_trial(
            float(p["Alpha_p"]), float(p["Alpha_n"]), len(d), float(p["Ev_Cut"]), float(p["Eff_intercept"]),
            trialtype, pd.to_numeric(d["point_amount"], errors="coerce"), pd.to_numeric(d["estimated_points"], errors="coerce")
        )
        obs = pd.to_numeric(d["Eff_Rating"], errors="coerce").dropna().to_numpy()
        n = min(len(pred), len(obs))
        for i in range(n):
            rows.append({"ID": pid, "rating_index": i + 1, "predicted": pred[i], "observed": obs[i], "residual": obs[i] - pred[i]})

    fit_df = pd.DataFrame(rows)
    if fit_df.empty:
        return fit_df
    rmse = fit_df.groupby("ID").apply(lambda d: np.sqrt(np.mean((d["observed"] - d["predicted"]) ** 2))).rename("RMSE").reset_index()
    return fit_df.merge(rmse, on="ID", how="left")

# =============================================================================
# MERGE SELF-REPORT/DESCRIPTIVES
# =============================================================================


def extract_finalq_from_raw(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Extract post-task FinalQ items from raw behavioral data_May_DC8.csv.

    The raw task file is row/trial-level, but FinalQ items appear only on one/few rows
    per participant. This function returns a participant-level table with exactly one
    row per ID so it can be safely merged into final_df.
    """
    finalq_cols = ["FinalQ_1", "FinalQ_2", "FinalQ_3", "FinalQ_4"]
    missing = [c for c in finalq_cols if c not in raw_data.columns]
    if missing:
        raise KeyError(
            f"Raw data is missing FinalQ columns: {missing}. "
            f"Available columns are: {raw_data.columns.tolist()}"
        )

    fq = raw_data[["ID"] + finalq_cols].copy()
    fq["ID"] = fq["ID"].astype(str).str.strip()

    # Keep only real participant IDs and rows where at least one FinalQ response exists.
    fq = fq[(fq["ID"].notna()) & (fq["ID"] != "") & (fq["ID"].str.lower() != "nan")].copy()
    fq = fq.dropna(subset=finalq_cols, how="all").copy()

    for c in finalq_cols:
        fq[c] = pd.to_numeric(fq[c], errors="coerce")

    # Keep rows where all four responses are present and numeric.
    fq = fq.dropna(subset=finalq_cols, how="any").copy()

    if fq.empty:
        raise ValueError(
            "No complete FinalQ rows were found in raw_data. "
            "Check whether FinalQ items are stored under different column names."
        )

    # If multiple complete rows exist per participant, keep the first complete response.
    n_before = len(fq)
    n_dup_ids = fq["ID"].duplicated().sum()
    fq = fq.sort_values("ID").drop_duplicates(subset="ID", keep="first").reset_index(drop=True)

    assert_one_row_per_participant(fq, "FinalQ table extracted from raw_data")
    print("\n=== Extracted FinalQ items from raw_data ===", flush=True)
    print(f"Complete FinalQ rows before ID collapse: {n_before}", flush=True)
    print(f"Duplicate complete FinalQ rows removed: {n_dup_ids}", flush=True)
    print(f"FinalQ participant rows: {len(fq)}", flush=True)
    print(fq[finalq_cols].describe(), flush=True)
    return fq

def load_questionnaires_and_demographics(questionnaire_path: Path = QUESTIONNAIRE_PATH, demographics_path: Path = DEMOGRAPHICS_PATH):
    questionnaires = pd.read_csv(questionnaire_path)
    demographics = pd.read_csv(demographics_path)
    questionnaires["ID"] = questionnaires["ID"].astype(str).str.strip()
    possible_id_cols = ["ID", "Participant id",  "Subject"]

    id_col = next((col for col in possible_id_cols if col in demographics.columns), None)

    if id_col is None:
        raise KeyError(
            "Could not find an ID column in demographics. "
            f"Available columns are: {demographics.columns.tolist()}"
        )

    demographics = demographics.rename(columns={id_col: "ID"})
    demographics["ID"] = demographics["ID"].astype(str).str.strip()   

    # Keep all columns but compute known subscales when available.
    rrs_brooding = ["RRS_5", "RRS_10", "RRS_13", "RRS_15", "RRS_16"]
    rrs_reflection = ["RRS_7", "RRS_11", "RRS_12", "RRS_20", "RRS_21"]
    rrs_depression = ["RRS_1", "RRS_2", "RRS_3", "RRS_4", "RRS_6", "RRS_8", "RRS_9", "RRS_14", "RRS_17", "RRS_18", "RRS_19", "RRS_22"]
    dars_consum = ["A2_1", "A2_2", "A2_3", "A2_4"]
    dars_effort = ["B2_1", "B2_2", "B2_3", "B2_4"]
    dars_motiv = ["C2_1", "C2_2", "C2_3", "C2_4"]
    dars_interest = ["D2_1", "D2_2", "D2_3", "D2_4", "D2_5"]

    all_subscale_cols = rrs_brooding + rrs_reflection + rrs_depression + dars_consum + dars_effort + dars_motiv + dars_interest
    questionnaires = safe_numeric(questionnaires, all_subscale_cols + ["RRS_Score", "PTQ_Score", "DARS_Score", "PSWQ_Score"])
    if set(rrs_brooding).issubset(questionnaires.columns):
        questionnaires["RRS_Brooding"] = questionnaires[rrs_brooding].sum(axis=1, min_count=1)
    if set(rrs_reflection).issubset(questionnaires.columns):
        questionnaires["RRS_Reflection"] = questionnaires[rrs_reflection].sum(axis=1, min_count=1)
    if set(rrs_depression).issubset(questionnaires.columns):
        questionnaires["RRS_Depression"] = questionnaires[rrs_depression].sum(axis=1, min_count=1)
    if set(dars_consum).issubset(questionnaires.columns):
        questionnaires["DARS_Consum"] = questionnaires[dars_consum].sum(axis=1, min_count=1)
    if set(dars_effort).issubset(questionnaires.columns):
        questionnaires["DARS_Effort"] = questionnaires[dars_effort].sum(axis=1, min_count=1)
    if set(dars_motiv).issubset(questionnaires.columns):
        questionnaires["DARS_Motiv"] = questionnaires[dars_motiv].sum(axis=1, min_count=1)
    if set(dars_interest).issubset(questionnaires.columns):
        questionnaires["DARS_Interest"] = questionnaires[dars_interest].sum(axis=1, min_count=1)
    return questionnaires, demographics


def make_final_df(
    questionnaires: pd.DataFrame,
    demographics: pd.DataFrame,
    metrics: pd.DataFrame,
    finalq: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the participant-level analysis dataframe.

    This function should only merge/calculate participant-level variables. It should not run
    statistical models or plotting code. That keeps downstream analyses from accidentally
    using a trial-level dataframe.
    """
    q = collapse_questionnaires(questionnaires)
    d = collapse_one_row(demographics)
    m = collapse_one_row(metrics)

    assert_one_row_per_participant(q, "questionnaires collapsed")
    assert_one_row_per_participant(d, "demographics collapsed")
    assert_one_row_per_participant(m, "metrics collapsed")

    final_df = reduce(
        lambda L, R: L.merge(R, on="ID", how="inner", validate="one_to_one"),
        [d, q, m],
    )

    if finalq is not None:
        fq = collapse_one_row(finalq)
        assert_one_row_per_participant(fq, "FinalQ collapsed")
        # FinalQ items are post-task/ecological-validity variables. They should
        # not determine the primary task-analysis sample.
        final_df = final_df.merge(fq, on="ID", how="left", validate="one_to_one")

    final_df = safe_numeric(final_df, [
        "Age", "RRS_Score", "PTQ_Score", "DARS_Score", "PSWQ_Score",
        "Alpha_p", "Alpha_n", "Ev_Cut", "Eff_intercept", "AIC", "SSE",
        "sse_total", "sse_low", "sse_high", "sse_diff",
        "avg_eff_rating_high", "avg_eff_rating_low",
        "avg_diff_eff_minus_true", "avg_diff_eff_minus_true_high", "avg_diff_eff_minus_true_low",
        "avg_condition_order", "alpha_diff",
        "FinalQ_1", "FinalQ_2", "FinalQ_3", "FinalQ_4",
    ])

    # Group definitions used throughout the manuscript.
    conditions = [
        final_df["RRS_Score"] <= 40,
        (final_df["RRS_Score"] >= 60) & (final_df["DARS_Score"] <= 44),
        (final_df["RRS_Score"] >= 60) & (final_df["DARS_Score"] > 44),
    ]
    labels_raw = ["low RRS", "high RRS, high Anh", "high RRS, low Anh"]
    final_df["Group"] = np.select(conditions, labels_raw, default="Other")
    final_df["Group_plot"] = final_df["Group"].map(RAW_GROUP_TO_PLOT).fillna(final_df["Group"])
    final_df["Group_plot"] = pd.Categorical(
        final_df["Group_plot"],
        categories=["LRLA", "HRHA", "HRLA", "Other"],
        ordered=True,
    )

    if REQUIRE_COMPLETE_CORE_COVARIATES:
        incomplete_mask = final_df[CORE_ANALYSIS_COLUMNS].isna().any(axis=1)
        excluded = final_df.loc[incomplete_mask, ["ID", "Age", "Sex", "RRS_Score", "DARS_Score", "avg_condition_order", "Group", "Group_plot"]].copy()
        if not excluded.empty:
            excluded.to_csv(OUTPUT_DIR / "excluded_incomplete_core_covariates.csv", index=False)
            print("\n=== Excluding participants missing core analysis covariates ===")
            print(excluded.to_string(index=False))
        final_df = final_df.loc[~incomplete_mask].copy()

    if EXCLUDE_OTHER_GROUP:
        other_mask = final_df["Group_plot"].astype(str).eq("Other")
        excluded = final_df.loc[other_mask, ["ID", "RRS_Score", "DARS_Score", "Group", "Group_plot"]].copy()
        if not excluded.empty:
            excluded.to_csv(OUTPUT_DIR / "excluded_other_group_participants.csv", index=False)
            print("\n=== Excluding participants classified as Other ===")
            print(excluded.to_string(index=False))
        final_df = final_df.loc[~other_mask].copy()
        final_df["Group_plot"] = pd.Categorical(
            final_df["Group_plot"].astype(str),
            categories=GROUP_ORDER,
            ordered=True,
        )

    # Convenience aliases for learning-rate analyses.
    final_df["ThreeGroup"] = final_df["Group_plot"].astype(str)
    if "Alpha_p" in final_df.columns and "Alpha_n" in final_df.columns:
        final_df["Alpha_p_minus_Alpha_n"] = final_df["Alpha_p"] - final_df["Alpha_n"]
        final_df["Alpha_abs_diff"] = (final_df["Alpha_p"] - final_df["Alpha_n"]).abs()

    assert_one_row_per_participant(final_df, "final_df")
    print("\n=== Final merged dataset ===")
    print("N rows:", len(final_df))
    print("Unique IDs:", final_df["ID"].nunique())
    print(final_df["Group_plot"].value_counts(dropna=False))
    return final_df


def compare_learning_rates_by_group(final_df: pd.DataFrame, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare Alpha_p/Alpha_n across LRLA, HRHA, and HRLA using participant-level data.

    IMPORTANT: This function intentionally takes final_df only. Learning rates are fitted
    participant-level model parameters, so the analysis must use exactly one row per ID.
    """
    assert_one_row_per_participant(final_df, "final_df entering learning-rate comparisons")
    print("\nSTARTING participant-level learning-rate group comparisons...", flush=True)

    lr_df = final_df.copy()
    lr_df["ID"] = lr_df["ID"].astype(str).str.strip()
    lr_df = lr_df[lr_df["Group_plot"].astype(str).isin(GROUP_ORDER)].copy()

    for col in ["Alpha_p", "Alpha_n", "Age", "avg_condition_order"]:
        if col in lr_df.columns:
            lr_df[col] = pd.to_numeric(lr_df[col], errors="coerce")

    lr_df["Alpha_p_minus_Alpha_n"] = lr_df["Alpha_p"] - lr_df["Alpha_n"]
    lr_df["Alpha_abs_diff"] = (lr_df["Alpha_p"] - lr_df["Alpha_n"]).abs()

    participant_cols = [
        "ID", "Group_plot", "Age", "Sex", "avg_condition_order",
        "Alpha_p", "Alpha_n", "Alpha_p_minus_Alpha_n", "Alpha_abs_diff",
    ]
    participant_cols = [c for c in participant_cols if c in lr_df.columns]
    lr_df = lr_df[participant_cols].copy()
    assert_one_row_per_participant(lr_df, "learning-rate dataframe")
    print("Learning-rate group counts:")
    print(lr_df["Group_plot"].value_counts(dropna=False), flush=True)

    outcomes = ["Alpha_p", "Alpha_n", "Alpha_p_minus_Alpha_n", "Alpha_abs_diff"]
    anova_rows = []
    pairwise_rows = []

    for outcome in outcomes:
        print(f"Running learning-rate group model for {outcome}...", flush=True)
        needed = [outcome, "Group_plot", "Age", "Sex"]
        if "avg_condition_order" in lr_df.columns:
            needed.append("avg_condition_order")
        d = lr_df.dropna(subset=needed).copy()
        d["Group_plot"] = pd.Categorical(d["Group_plot"].astype(str), categories=GROUP_ORDER, ordered=True)

        covariates = "Age + C(Sex)"
        if "avg_condition_order" in d.columns:
            covariates += " + avg_condition_order"
        formula = f"{outcome} ~ C(Group_plot) + {covariates}"
        model = smf.ols(formula, data=d).fit()
        aov = anova_lm(model, typ=2)

        if "C(Group_plot)" in aov.index:
            anova_rows.append({
                "outcome": outcome,
                "N": int(model.nobs),
                "df_group": aov.loc["C(Group_plot)", "df"],
                "df_resid": model.df_resid,
                "F_group": aov.loc["C(Group_plot)", "F"],
                "p_group": aov.loc["C(Group_plot)", "PR(>F)"],
                "formula": formula,
            })

        # Pairwise models: simple and robust against contrast-rank warnings.
        for g1, g2 in combinations(GROUP_ORDER, 2):
            sub = d[d["Group_plot"].astype(str).isin([g1, g2])].copy()
            sub["pair_group"] = (sub["Group_plot"].astype(str) == g2).astype(int)
            pair_formula = f"{outcome} ~ pair_group + {covariates}"
            pair_model = smf.ols(pair_formula, data=sub).fit()
            pairwise_rows.append({
                "outcome": outcome,
                "contrast": f"{g1} - {g2}",
                # estimate is g2-g1 because pair_group=1 for g2; multiply by -1 to label g1-g2.
                "estimate": -pair_model.params.get("pair_group", np.nan),
                "t": -pair_model.tvalues.get("pair_group", np.nan),
                "p_raw": pair_model.pvalues.get("pair_group", np.nan),
                "N": int(pair_model.nobs),
                "formula": pair_formula,
            })
        print(f"Finished learning-rate group model for {outcome}.", flush=True)

    anova_df = pd.DataFrame(anova_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df["p_holm"] = np.nan
        for outcome in outcomes:
            mask = pairwise_df["outcome"].eq(outcome)
            if mask.sum() > 0:
                pairwise_df.loc[mask, "p_holm"] = multipletests(pairwise_df.loc[mask, "p_raw"], method="holm")[1]

    print("\n=== Learning-rate ANCOVA summary ===")
    print(anova_df.to_string(index=False))
    print("\n=== Learning-rate pairwise summary ===")
    print(pairwise_df.to_string(index=False))

    if save:
        anova_df.to_csv(OUTPUT_DIR / "learning_rate_group_ANCOVA_summary.csv", index=False)
        pairwise_df.to_csv(OUTPUT_DIR / "learning_rate_group_pairwise_summary.csv", index=False)

        # Save, do not show: avoids terminal runs getting stuck at a hidden Matplotlib window.
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
        plot_specs = [
            ("Alpha_p", "Positive learning rate"),
            ("Alpha_n", "Negative learning rate"),
            ("Alpha_p_minus_Alpha_n", "Learning-rate asymmetry\nAlpha_p − Alpha_n"),
        ]
        for ax, (outcome, title) in zip(axes, plot_specs):
            d = lr_df.dropna(subset=[outcome, "Group_plot"]).copy()
            stats_df = d.groupby("Group_plot", observed=True)[outcome].agg(["mean", "std", "count"]).reindex(GROUP_ORDER)
            stats_df["se"] = stats_df["std"] / np.sqrt(stats_df["count"])
            x = np.arange(len(GROUP_ORDER))
            ax.bar(x, stats_df["mean"].values, width=0.7)
            ax.errorbar(x, stats_df["mean"].values, yerr=stats_df["se"].values, fmt="none", capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(GROUP_ORDER)
            ax.set_title(title)
            ax.set_ylabel("Participant mean")
            beautify_axes(ax)
        fig.suptitle("Learning-rate parameters by group", y=1.05)
        fig.savefig(OUTPUT_DIR / "learning_rate_group_comparisons.png", dpi=600, bbox_inches="tight")
        plt.close()

    print("FINISHED participant-level learning-rate group comparisons.\n", flush=True)
    return anova_df, pairwise_df

# =============================================================================
# RESULTS: TASK ACCURACY AND GROUP EFFECTS
# =============================================================================

def run_group_ancova(df: pd.DataFrame, dv: str, include_order: bool = True):
    dat = df[df["Group_plot"].isin(GROUP_ORDER)].copy()
    needed = [dv, "Group_plot", "Age", "Sex"]
    if include_order and "avg_condition_order" in dat.columns:
        needed.append("avg_condition_order")
    dat = dat.dropna(subset=needed)
    cov = "Age + C(Sex)"
    if include_order and "avg_condition_order" in dat.columns:
        cov += " + avg_condition_order"
    formula = f"{dv} ~ C(Group_plot) + {cov}"
    model = smf.ols(formula, data=dat).fit()
    aov = anova_lm(model, typ=2)
    return model, aov


def run_pairwise_group_contrasts(df: pd.DataFrame, dv: str, include_order: bool = True) -> pd.DataFrame:
    rows = []
    for g1, g2 in combinations(GROUP_ORDER, 2):
        sub = df[df["Group_plot"].isin([g1, g2])].copy()
        sub["contrast_group"] = (sub["Group_plot"].astype(str) == g2).astype(int)
        needed = [dv, "contrast_group", "Age", "Sex"]
        if include_order and "avg_condition_order" in sub.columns:
            needed.append("avg_condition_order")
        sub = sub.dropna(subset=needed)
        cov = "Age + C(Sex)"
        if include_order and "avg_condition_order" in sub.columns:
            cov += " + avg_condition_order"
        m = smf.ols(f"{dv} ~ contrast_group + {cov}", data=sub).fit()
        rows.append({
            "DV": dv, "contrast": f"{g1} - {g2}", "b": m.params.get("contrast_group", np.nan),
            "t": m.tvalues.get("contrast_group", np.nan), "p": m.pvalues.get("contrast_group", np.nan),
            "N": int(m.nobs),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = multipletests(out["p"], method="holm")[1]
        out["stars"] = out["p_holm"].apply(p_to_stars)
    return out



def figure1_task_performance_by_group(final_df: pd.DataFrame, save=True):
    """Manuscript Figure 2: task performance by group.

    This intentionally reproduces the manuscript styling: three violin panels,
    log-transformed SSE outcomes, individual jittered observations, adjusted
    means with bootstrapped 95% CIs, and the LRLA-HRLA-HRHA order.
    """
    df = final_df.copy()
    raw_y_cols = ["sse_total", "sse_low", "sse_high"]
    y_cols = ["log_sse_total", "log_sse_low", "log_sse_high"]
    titles = {
        "log_sse_total": "Full-task accuracy error",
        "log_sse_low": "Low-efficacy condition accuracy error",
        "log_sse_high": "High-efficacy condition accuracy error",
    }
    y_label = "log-transformed SSE (lower = better)"
    for raw_col in raw_y_cols:
        df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")
        df[f"log_{raw_col}"] = np.log1p(df[raw_col])

    group_order = FIGURE_GROUP_ORDER
    df = df[df["Group_plot"].astype(str).isin(group_order)].copy()
    df["Group_plot"] = pd.Categorical(df["Group_plot"].astype(str), categories=group_order, ordered=True)

    def emm_and_ci_bootstrap(model, df_design, group_levels, group_col, n_boot=3000, seed=123):
        rng = np.random.default_rng(seed)
        emms = {}
        for g in group_levels:
            tmp = df_design.copy()
            tmp[group_col] = g
            emms[g] = float(model.predict(tmp).mean())
        boots = {g: np.empty(n_boot, dtype=float) for g in group_levels}
        n = len(df_design)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_df = df_design.iloc[idx].copy()
            for g in group_levels:
                tmp = boot_df.copy()
                tmp[group_col] = g
                boots[g][b] = float(model.predict(tmp).mean())
        cis = {g: tuple(np.quantile(boots[g], [0.025, 0.975]).astype(float)) for g in group_levels}
        return emms, cis

    def p_to_stars_local(p):
        if pd.isna(p):
            return ""
        if p < .001:
            return "***"
        if p < .01:
            return "**"
        if p < .05:
            return "*"
        return ""

    def prereg_contrasts(model, group_col="Group_plot"):
        params = list(model.params.index)
        def coef_name(g):
            return f"C({group_col})[T.{g}]"
        rows = []
        # C1: HRLA vs mean(LRLA, HRHA)
        L = np.zeros((1, len(params)))
        if coef_name("HRLA") in params:
            L[0, params.index(coef_name("HRLA"))] = 1.0
        if coef_name("HRHA") in params:
            L[0, params.index(coef_name("HRHA"))] = -0.5
        rows.append({"contrast": "HRLA vs mean(LRLA, HRHA)", "p": float(np.asarray(model.t_test(L).pvalue).squeeze())})
        # C2: HRHA vs LRLA
        L = np.zeros((1, len(params)))
        if coef_name("HRHA") in params:
            L[0, params.index(coef_name("HRHA"))] = 1.0
        rows.append({"contrast": "HRHA vs LRLA", "p": float(np.asarray(model.t_test(L).pvalue).squeeze())})
        out = pd.DataFrame(rows)
        out["p_adj"] = multipletests(out["p"], method="holm")[1]
        out["stars"] = out["p_adj"].apply(p_to_stars_local)
        return out

    fig, axes = plt.subplots(1, 3, figsize=(17.8, 7.4), sharex=False, sharey=False)
    stats_rows = []
    rng_base = 123
    for k, (ax, y) in enumerate(zip(axes, y_cols)):
        dat = df.dropna(subset=[y, "Group_plot", "Age", "Sex", "avg_condition_order"]).copy()
        formula = f"{y} ~ C(Group_plot) + Age + C(Sex) + avg_condition_order"
        model = smf.ols(formula, data=dat).fit()
        aov = anova_lm(model, typ=2)
        emms, cis = emm_and_ci_bootstrap(model, dat, group_order, "Group_plot", n_boot=3000, seed=rng_base + k)

        data_by_group = [dat.loc[dat["Group_plot"].astype(str).eq(g), y].values for g in group_order]
        positions = np.arange(len(group_order))
        viol = ax.violinplot(data_by_group, positions=positions, widths=0.85, showmeans=False, showmedians=True, showextrema=False)
        for body, g in zip(viol["bodies"], group_order):
            body.set_facecolor(GROUP_COLORS[g])
            body.set_edgecolor(GROUP_COLORS[g])
            body.set_alpha(0.35)
        if "cmedians" in viol:
            viol["cmedians"].set_color("black")
            viol["cmedians"].set_linewidth(1.3)

        rng = np.random.default_rng(rng_base + k)
        for i, g in enumerate(group_order):
            yvals = dat.loc[dat["Group_plot"].astype(str).eq(g), y].values
            xvals = rng.uniform(i - 0.11, i + 0.11, size=len(yvals))
            ax.scatter(xvals, yvals, s=18, alpha=0.35, color="#D79B4D", edgecolors="none")

        for i, g in enumerate(group_order):
            mu = emms[g]
            lo, hi = cis[g]
            ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], fmt="D", color="black", ecolor="black", capsize=4, linewidth=1.8, markersize=6, zorder=6)

        counts = dat.groupby("Group_plot", observed=True)[y].size().reindex(group_order).fillna(0).astype(int)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{g}\n(n={counts.loc[g]})" for g in group_order])
        ax.set_xlabel("Group")
        ax.set_ylabel(y_label)
        ax.set_title(titles[y], fontsize=18)
        ax.tick_params(axis="both", labelsize=15)
        ax.yaxis.label.set_size(18)
        ax.xaxis.label.set_size(18)
        beautify_axes(ax)

        cons = prereg_contrasts(model)
        top = np.nanmax(dat[y].values)
        bottom = np.nanmin(dat[y].values)
        yr = top - bottom if np.isfinite(top - bottom) and (top - bottom) > 0 else 1.0
        current = top + 0.05 * yr
        h = 0.025 * yr
        step = 0.08 * yr
        c1 = cons.iloc[0]
        c2 = cons.iloc[1]
        if c1["stars"]:
            ax.plot([0, 0, 2, 2], [current, current+h, current+h, current], color="black", linewidth=1.4)
            ax.text(1, current+h, c1["stars"], ha="center", va="bottom", fontsize=16)
            ax.scatter([1], [current+h], marker="v", s=85, color="black", zorder=8)
            current += step
        if c2["stars"]:
            ax.plot([0, 0, 2, 2], [current, current+h, current+h, current], color="black", linewidth=1.4)
            ax.text(1, current+h, c2["stars"], ha="center", va="bottom", fontsize=16)
            current += step
        if c1["stars"] or c2["stars"]:
            ax.set_ylim(bottom - 0.05*yr, current + 0.05*yr)

        stats_rows.append({"DV": y, "F": aov.loc["C(Group_plot)", "F"], "df1": aov.loc["C(Group_plot)", "df"], "df2": model.df_resid, "p": aov.loc["C(Group_plot)", "PR(>F)"]})

    handles = [
        Line2D([0], [0], marker="D", color="black", linestyle="None", markersize=10, label="Age/Sex-adjusted mean (EMM) ± 95% CI"),
        Line2D([0], [0], marker="v", color="black", linestyle="None", markersize=8, label="Central point in contrast: HRLA > (LRLA and HRHA)"),
        Line2D([0], [0], linestyle="None", label="Significance: * p < .05, ** p < .01, *** p < .001 (holm-corrected, OLS)"),
        Line2D([0], [0], linestyle="None", label="LRLA = Low rumination / Low anhedonia"),
        Line2D([0], [0], linestyle="None", label="HRLA = High rumination / Low anhedonia"),
        Line2D([0], [0], linestyle="None", label="HRHA = High rumination / High anhedonia"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06, 0, 1), ncol=2, frameon=False, fontsize=16)
    plt.tight_layout(rect=[0.0, 0.1, 1.0, 1.0])
    if save:
        fig.savefig(OUTPUT_DIR / "figure2_task_performance_by_group.png", dpi=600, bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / "Figure2_Task_Performance_by_Group.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    out = pd.DataFrame(stats_rows)
    print("\n=== Manuscript Figure 2 ANCOVAs ===")
    print(out.to_string(index=False))
    return out


def figure2_condition_difference_regressions(final_df: pd.DataFrame, save=True):
    """Manuscript Figure 3: residualized condition-difference regressions."""
    df = final_df.copy()
    dv = "sse_diff"
    needed = [dv, "DARS_Score", "RRS_Score", "Age", "Sex", "avg_condition_order"]
    dat = df.dropna(subset=[c for c in needed if c in df.columns]).copy()

    formula = f"{dv} ~ DARS_Score + RRS_Score + Age + C(Sex) + avg_condition_order"
    model = smf.ols(formula, data=dat).fit()
    model_std_dat = dat.copy()
    for c in [dv, "DARS_Score", "RRS_Score", "Age"]:
        model_std_dat[f"{c}_z"] = safe_z(model_std_dat[c])
    model_std = smf.ols(f"{dv}_z ~ DARS_Score_z + RRS_Score_z + Age_z + C(Sex) + avg_condition_order", data=model_std_dat).fit()

    print("\n=== Manuscript Figure 3 model: condition performance difference ===")
    print(model.summary())

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    specs = [
        ("DARS_Score", DARS_COLOR, "a", "DARS and efficacy accuracy difference", "Anhedonia (DARS) score\nControlling for RRS, Age, Sex, and Condition Order"),
        ("RRS_Score", RRS_COLOR, "b", "RRS and efficacy accuracy difference", "Rumination (RRS) score\nControlling for DARS, Age, Sex, and Condition Order"),
    ]
    for ax, (xcol, color, panel, title, xlabel) in zip(axes, specs):
        other = "RRS_Score" if xcol == "DARS_Score" else "DARS_Score"
        controls = [other, "Age", "C(Sex)", "avg_condition_order"]
        y_res = smf.ols(f"{dv} ~ " + " + ".join(controls), data=dat).fit().resid
        x_res = smf.ols(f"{xcol} ~ " + " + ".join(controls), data=dat).fit().resid
        ax.scatter(x_res, y_res, s=15, alpha=0.45, color=color, edgecolors="none")
        lr = stats.linregress(x_res, y_res)
        xs = np.linspace(np.nanmin(x_res), np.nanmax(x_res), 200)
        ax.plot(xs, lr.intercept + lr.slope * xs, lw=2.5, color=color)
        ax.fill_between(xs, lr.intercept + lr.slope * xs, lr.intercept + lr.slope * xs, color=color, alpha=0.15)
        ax.set_title(f"{panel}\n{title}", loc="left", fontweight="bold", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Efficacy accuracy difference high - low\nControlling for covariates")
        beta_name = f"{xcol}_z"
        beta = model_std.params.get(beta_name, np.nan)
        pval = model.pvalues.get(xcol, np.nan)
        label = "DARS" if xcol == "DARS_Score" else "RRS"
        ax.text(0.03, 0.97, f"{label} β* = {beta:.3f}, p = {pval:.4f}", transform=ax.transAxes, va="top", fontsize=9)
        beautify_axes(ax)
    if save:
        fig.savefig(OUTPUT_DIR / "figure3_condition_difference_regressions.png", dpi=600, bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / "Figure3_Condition_Difference_Regressions.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return model

# =============================================================================
# FIGURE 3: MODEL FIT
# =============================================================================

def binned_means(x, y, bins=20, x_min=0, x_max=1):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    edges = np.linspace(x_min, x_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    y_mean = np.full(bins, np.nan)
    y_se = np.full(bins, np.nan)
    for i in range(bins):
        mask = (x >= edges[i]) & (x <= edges[i + 1] if i == bins - 1 else x < edges[i + 1])
        if mask.sum() > 0:
            y_mean[i] = np.nanmean(y[mask])
        if mask.sum() > 1:
            y_se[i] = np.nanstd(y[mask], ddof=1) / np.sqrt(mask.sum())
    return centers, y_mean, y_se



def figure3_model_fit(fit_df: pd.DataFrame, final_df: pd.DataFrame, save=True):
    """Manuscript Figure 4: model calibration and RMSE by group."""
    fit_plot = fit_df.copy()
    fit_plot["ID"] = fit_plot["ID"].astype(str)
    group_df = final_df[["ID", "Group_plot"]].copy()
    group_df["ID"] = group_df["ID"].astype(str)
    group_df = group_df.drop_duplicates("ID")
    rmse_plot = fit_plot[["ID", "RMSE"]].drop_duplicates().merge(group_df, on="ID", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)

    ax1 = axes[0]
    beautify_axes(ax1)
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, alpha=0.8, color="black")
    centers, y_mean, y_se = binned_means(fit_plot["predicted"], fit_plot["observed"], bins=20, x_min=0, x_max=1)
    m = np.isfinite(y_mean)
    ax1.errorbar(centers[m], y_mean[m], yerr=y_se[m], fmt="o-", markersize=5, linewidth=3.2, capsize=3, color=MODEL_ORANGE, ecolor=MODEL_ORANGE, zorder=5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Model-predicted efficacy")
    ax1.set_ylabel("Observed efficacy")
    ax1.set_title("A  Calibration", loc="left", fontweight="bold")

    ax2 = axes[1]
    beautify_axes(ax2)
    tmp = rmse_plot[rmse_plot["Group_plot"].astype(str).isin(FIGURE_GROUP_ORDER)].copy()
    tmp["Group_plot"] = pd.Categorical(tmp["Group_plot"].astype(str), categories=FIGURE_GROUP_ORDER, ordered=True)
    stats_df = tmp.groupby("Group_plot", observed=True)["RMSE"].agg(["mean", "std", "count"]).reindex(FIGURE_GROUP_ORDER)
    stats_df["se"] = stats_df["std"] / np.sqrt(stats_df["count"])
    x = np.arange(len(FIGURE_GROUP_ORDER))
    ax2.bar(x, stats_df["mean"].values, width=0.7, color=MODEL_BLUE)
    ax2.errorbar(x, stats_df["mean"].values, yerr=stats_df["se"].values, fmt="none", capsize=3, linewidth=1.2, color="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(FIGURE_GROUP_ORDER)
    ax2.set_xlabel("Group")
    ax2.set_ylabel("RMSE")
    ax2.set_title("B  Participant RMSE by group", loc="left", fontweight="bold")

    if save:
        fig.savefig(OUTPUT_DIR / "figure4_model_fit.png", dpi=600, bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / "Figure4_Model_Fit.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    rmse_summary = fit_plot[["ID", "RMSE"]].drop_duplicates()["RMSE"].describe()
    print("\n=== Manuscript Figure 4 RMSE summary ===")
    print(rmse_summary)
    return rmse_summary

# =============================================================================
# FIGURE 4: LEARNING RATE ASSOCIATIONS
# =============================================================================

def fit_continuous_symptom_model(df: pd.DataFrame, dv: str, transform: str | None = None):
    dat = df.copy()
    ycol = dv
    if transform == "log":
        ycol = f"log_{dv}"
        dat[ycol] = np.log(pd.to_numeric(dat[dv], errors="coerce") + 1e-8)
    elif transform == "yeojohnson":
        ycol = f"yj_{dv}"
        vals = pd.to_numeric(dat[dv], errors="coerce")
        notna = vals.notna()
        dat.loc[notna, ycol], _ = yeojohnson(vals.loc[notna])

    needed = [ycol, "DARS_Score", "RRS_Score", "Age", "Sex", "avg_condition_order"]
    dat = dat.dropna(subset=[c for c in needed if c in dat.columns]).copy()
    dat[f"{ycol}_z"] = safe_z(dat[ycol])
    dat["DARS_Score_z"] = safe_z(dat["DARS_Score"])
    dat["RRS_Score_z"] = safe_z(dat["RRS_Score"])
    dat["Age_z"] = safe_z(dat["Age"])
    formula = f"{ycol}_z ~ DARS_Score_z + RRS_Score_z + Age_z + C(Sex) + avg_condition_order"
    return smf.ols(formula, data=dat).fit(), dat, ycol



def partial_plot(ax, data, dv, focal, control_label, title, color, annotation_model=None, annotation_model_std=None):
    controls = ["Age", "C(Sex)", "avg_condition_order", "RRS_Score" if focal == "DARS_Score" else "DARS_Score"]
    y_res = smf.ols(f"{dv} ~ " + " + ".join(controls), data=data).fit().resid
    x_res = smf.ols(f"{focal} ~ " + " + ".join(controls), data=data).fit().resid
    ax.scatter(x_res, y_res, s=15, alpha=0.42, color=color, edgecolors="none")
    lr = stats.linregress(x_res, y_res)
    xs = np.linspace(np.nanmin(x_res), np.nanmax(x_res), 200)
    ax.plot(xs, lr.intercept + lr.slope * xs, lw=2.5, color=color)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
    xlab = "Anhedonia (DARS) score" if focal == "DARS_Score" else "Rumination (RRS) score"
    ax.set_xlabel(f"{xlab}\nControlling for {control_label}, Age, Sex, and Condition Order", fontsize=7)
    ax.set_ylabel(f"{PRETTY_OUTCOME.get(dv, dv)}\nControlling for covariates", fontsize=7)
    if annotation_model is not None and annotation_model_std is not None:
        beta_name = f"{focal}_z"
        label = "DARS" if focal == "DARS_Score" else "RRS"
        beta = annotation_model_std.params.get(beta_name, np.nan)
        p = annotation_model.pvalues.get(focal, np.nan)
        ax.text(0.03, 0.97, f"{label} β* = {beta:.3f}, p = {p:.4f}", transform=ax.transAxes, va="top", fontsize=6)
    beautify_axes(ax)


def figure4_learning_rate_associations(final_df: pd.DataFrame, save=True):
    """Manuscript Figure 5: learning-rate associations with DARS/RRS."""
    df = final_df.copy()
    results = []
    models = {}
    std_models = {}
    for dv in ["Alpha_p", "Alpha_n", "alpha_diff"]:
        m, dat, ycol = fit_continuous_symptom_model(df, dv, transform="yeojohnson" if dv in ["Alpha_p", "Alpha_n"] else None)
        models[dv] = (m, dat, ycol)
        std_models[dv] = m
        results.append({
            "DV": dv,
            "DARS_beta_std": m.params.get("DARS_Score_z", np.nan),
            "DARS_p": m.pvalues.get("DARS_Score_z", np.nan),
            "RRS_beta_std": m.params.get("RRS_Score_z", np.nan),
            "RRS_p": m.pvalues.get("RRS_Score_z", np.nan),
            "N": int(m.nobs),
            "formula": m.model.formula,
        })
    res = pd.DataFrame(results)
    print("\n=== Manuscript Figure 5 learning-rate models ===")
    print(res.to_string(index=False))

    fig, axes = plt.subplots(3, 2, figsize=(8.0, 7.35), constrained_layout=True)
    row_titles = {
        "Alpha_p": "Positive learning rate",
        "Alpha_n": "Negative learning rate",
        "alpha_diff": "Learning rate asymmetry",
    }
    plot_specs = [
        ("Alpha_p", "DARS_Score", "RRS", "a", DARS_COLOR),
        ("Alpha_p", "RRS_Score", "DARS", "b", RRS_COLOR),
        ("Alpha_n", "DARS_Score", "RRS", "c", DARS_COLOR),
        ("Alpha_n", "RRS_Score", "DARS", "d", RRS_COLOR),
        ("alpha_diff", "DARS_Score", "RRS", "e", DARS_COLOR),
        ("alpha_diff", "RRS_Score", "DARS", "f", RRS_COLOR),
    ]
    for ax, (dv, focal, control, panel, color) in zip(axes.ravel(), plot_specs):
        dat = df.dropna(subset=[dv, focal, "Age", "Sex", "avg_condition_order", "RRS_Score" if focal == "DARS_Score" else "DARS_Score"]).copy()
        partial_plot(ax, dat, dv, focal, control, f"{panel}\n{row_titles[dv]}", color, annotation_model=models[dv][0], annotation_model_std=std_models[dv])
    if save:
        fig.savefig(OUTPUT_DIR / "figure5_learning_rate_associations.png", dpi=600, bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / "Figure5_Learning_Rate_Associations.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return res

# =============================================================================
# FIGURE 5: FINALQ FACTOR ANALYSIS AND ASSOCIATIONS
# =============================================================================


def run_finalq_factor_analysis(final_df: pd.DataFrame, save=True):
    """Manuscript Figure 6: latent FinalQ factor and associations."""
    if not HAS_FACTOR_ANALYZER:
        raise ImportError("factor_analyzer is not installed. Install with: pip install factor_analyzer")

    DF = final_df.copy()
    FINALQS = ["FinalQ_1", "FinalQ_2", "FinalQ_3", "FinalQ_4"]
    PREDICTORS = [
        "Alpha_n", "Alpha_p", "sse_total", "sse_low", "sse_high", "avg_eff_rating_high", "avg_eff_rating_low",
        "avg_diff_eff_minus_true", "avg_diff_eff_minus_true_high", "avg_diff_eff_minus_true_low", "Eff_intercept",
    ]
    for col in FINALQS + PREDICTORS + ["Age", "avg_condition_order", "DARS_Score", "RRS_Score"]:
        if col in DF.columns:
            DF[col] = pd.to_numeric(DF[col], errors="coerce")
    DF["Sex"] = DF["Sex"].astype("category")

    efa_df = DF[FINALQS].dropna().copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(efa_df)
    chi2, bartlett_p = calculate_bartlett_sphericity(efa_df)
    kmo_item, kmo_total = calculate_kmo(efa_df)

    fa = FactorAnalyzer(n_factors=1, rotation=None, method="ml")
    fa.fit(efa_df)
    load1 = pd.DataFrame(fa.loadings_, index=FINALQS, columns=["Factor1"])
    scores = fa.transform(efa_df)
    score_df = pd.DataFrame(scores, index=efa_df.index, columns=["FinalQ_Factor1"])
    DF_with_scores = DF.join(score_df)

    print("\n=== FinalQ factor analysis ===")
    print(f"N used = {len(efa_df)}")
    print(f"Bartlett chi2 = {chi2:.4f}, p = {bartlett_p:.6g}")
    print(f"KMO total = {kmo_total:.4f}")
    print("Loadings:")
    print(load1)
    print("Variance explained:")
    print(fa.get_factor_variance())

    def fit_single_model(outcome, predictor):
        use_order = predictor in PREDICTORS and "avg_condition_order" in DF_with_scores.columns
        needed = [outcome, predictor, "Age", "Sex"]
        if use_order:
            needed.append("avg_condition_order")
        dat = DF_with_scores[needed].dropna().copy()
        if dat.shape[0] < 20:
            return None
        formula = f"{outcome} ~ {predictor} + Age + C(Sex)" + (" + avg_condition_order" if use_order else "")
        m = smf.ols(formula, data=dat).fit()
        ci = m.conf_int().loc[predictor]
        return {"Outcome": outcome, "Predictor": predictor, "N": int(m.nobs), "used_condition_order": use_order,
                "b": m.params[predictor], "SE": m.bse[predictor], "t": m.tvalues[predictor], "p": m.pvalues[predictor],
                "CI_low": ci[0], "CI_high": ci[1], "Adj_R2": m.rsquared_adj, "formula": formula}

    rows = []
    for pred in PREDICTORS:
        if pred in DF_with_scores.columns:
            out = fit_single_model("FinalQ_Factor1", pred)
            if out is not None:
                rows.append(out)
    results = pd.DataFrame(rows)
    results["p_FDR"] = multipletests(results["p"], method=P_CORR_METHOD)[1]
    results["sig_FDR"] = results["p_FDR"] < ALPHA
    results["stars_FDR"] = results["p_FDR"].apply(p_to_stars)
    results = results.sort_values(["p_FDR", "p"]).reset_index(drop=True)
    print("\n=== FinalQ factor associations ===")
    print(results[["Outcome", "Predictor", "N", "b", "SE", "t", "p", "p_FDR", "stars_FDR", "Adj_R2"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.13), gridspec_kw={"width_ratios": [1.0, 1.65]}, constrained_layout=True)

    ax = axes[0]
    item_labels = {
        "FinalQ_1": "How aware are you of\nhow your actions\nimpact others?",
        "FinalQ_2": "Are you able to adjust\nyour behavior after\nreceiving feedback?",
        "FinalQ_3": "Do you unfairly blame\nyourself for things\nthat happen to you?",
        "FinalQ_4": "Do you unfairly blame\nothers/external\ncircumstances for\nthings that happen to you?",
    }
    load_plot = load1.reset_index()
    load_plot.columns = ["Item", "Loading"]
    load_plot["Item"] = load_plot["Item"].map(item_labels)
    load_plot = load_plot.sort_values("Loading")
    ax.barh(load_plot["Item"], load_plot["Loading"], color=PURPLE)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Factor loading")
    ax.set_title("A  Loading from the factor solution", loc="left", fontweight="bold")
    beautify_axes(ax)

    ax = axes[1]
    label_map = {
        "Alpha_n": "Non-causal\nlearning",
        "Alpha_p": "Causal\nlearning",
        "sse_total": "Task accuracy",
        "sse_low": "Low efficacy condition\ntask accuracy",
        "sse_high": "High efficacy condition\ntask accuracy",
        "avg_eff_rating_high": "Average efficacy estimates\nin high efficacy condition",
        "avg_eff_rating_low": "Average efficacy estimates\nin low efficacy condition",
        "avg_diff_eff_minus_true": "Average efficacy estimates\nacross the task",
        "avg_diff_eff_minus_true_high": "High efficacy condition\nsigned attribution error",
        "avg_diff_eff_minus_true_low": "Low efficacy condition\nsigned attribution error",
        "Eff_intercept": "Full-task learning",
    }
    coef_df = results.copy()
    coef_df["plot_label"] = coef_df["Predictor"].map(label_map).fillna(coef_df["Predictor"])
    # Match manuscript: strongest positive at top, strongest negative at bottom.
    coef_df = coef_df.sort_values("b", ascending=True)
    ypos = np.arange(len(coef_df))
    for i, (_, row) in enumerate(coef_df.iterrows()):
        color = PURPLE if row["sig_FDR"] else LIGHT_PURPLE
        ax.errorbar(row["b"], i, xerr=[[row["b"] - row["CI_low"]], [row["CI_high"] - row["b"]]], fmt="o", color=color, ecolor=color, capsize=3, markersize=4)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(coef_df["plot_label"])
    ax.set_xlabel("Regression coefficient")
    ax.set_title("B  Association with the task factor", loc="left", fontweight="bold")
    beautify_axes(ax)

    if save:
        fig.savefig(OUTPUT_DIR / "figure6_finalq_latent_factor_associations.png", dpi=600, bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / "Figure6_FinalQ_Latent_Factor_Associations.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return DF_with_scores, results, load1

# =============================================================================
# SUPPLEMENTAL FIGURE 1: FINALQ GROUP MEANS
# =============================================================================

def supplemental_finalq_group_means(final_df: pd.DataFrame, save=True):
    DF = final_df.copy()
    FINALQS = ["FinalQ_1", "FinalQ_2", "FinalQ_3", "FinalQ_4"]
    rows, posthoc_rows = [], []

    for q in FINALQS:
        dat = DF[DF["Group_plot"].isin(GROUP_ORDER)].dropna(subset=[q, "Group_plot", "Age", "Sex", "avg_condition_order"])
        model = smf.ols(f"{q} ~ C(Group_plot) + Age + C(Sex) + avg_condition_order", data=dat).fit()
        aov = anova_lm(model, typ=2)
        rows.append({"DV": q, "F_group": aov.loc["C(Group_plot)", "F"], "df_group": aov.loc["C(Group_plot)", "df"],
                     "df_resid": model.df_resid, "p_group": aov.loc["C(Group_plot)", "PR(>F)"]})

        for g1, g2 in combinations(GROUP_ORDER, 2):
            sub = dat[dat["Group_plot"].isin([g1, g2])].copy()
            sub["pair"] = (sub["Group_plot"].astype(str) == g2).astype(int)
            m = smf.ols(f"{q} ~ pair + Age + C(Sex) + avg_condition_order", data=sub).fit()
            posthoc_rows.append({"DV": q, "contrast": f"{g1} - {g2}", "p": m.pvalues.get("pair", np.nan), "t": m.tvalues.get("pair", np.nan)})

    omni = pd.DataFrame(rows)
    omni["p_group_FDR"] = multipletests(omni["p_group"], method="fdr_bh")[1]
    posthoc = pd.DataFrame(posthoc_rows)
    posthoc["p_adj_withinDV"] = posthoc.groupby("DV")["p"].transform(lambda x: multipletests(x, method="holm")[1])
    posthoc["sig_pairwise"] = posthoc["p_adj_withinDV"] < 0.05

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    for ax, q in zip(axes.ravel(), FINALQS):
        dat = DF[DF["Group_plot"].isin(GROUP_ORDER)].dropna(subset=[q, "Group_plot"])
        means = dat.groupby("Group_plot", observed=True)[q].mean().reindex(GROUP_ORDER)
        sems = dat.groupby("Group_plot", observed=True)[q].sem().reindex(GROUP_ORDER)
        x = np.arange(len(GROUP_ORDER))
        ax.bar(x, means.values, yerr=sems.values, capsize=3, width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_title(q + ": " + FINALQ_LABELS[q], fontsize=9)
        beautify_axes(ax)
        row = omni[omni["DV"].eq(q)].iloc[0]
        ax.text(0.02, 0.98, f"F({int(row.df_group)}, {int(row.df_resid)}) = {row.F_group:.2f}\nFDR p = {row.p_group_FDR:.3g}", transform=ax.transAxes, va="top")

    plt.close()
    print("\n=== Supplemental Figure 1 FinalQ omnibus ===")
    print(omni.to_string(index=False))
    return omni, posthoc

# =============================================================================
# FINALQ ITEM-LEVEL PREDICTION FIGURE/RESULTS
# =============================================================================

def finalq_item_level_models(final_df: pd.DataFrame, save=True):
    DF = final_df.copy()
    FINALQS = ["FinalQ_1", "FinalQ_2", "FinalQ_3", "FinalQ_4"]
    OUTCOMES = [
        "avg_diff_eff_minus_true", "avg_diff_eff_minus_true_high", "avg_diff_eff_minus_true_low",
        "Alpha_p", "Alpha_n", "sse_total",
    ]
    rows = []
    for finalq in FINALQS:
        for outcome in OUTCOMES:
            needed = [outcome, finalq, "DARS_Score", "RRS_Score", "Age", "Sex", "avg_condition_order"]
            dat = DF.dropna(subset=[c for c in needed if c in DF.columns]).copy()
            # Corrected: these models control for symptoms and covariates; they do NOT control for Group.
            formula = f"{outcome} ~ {finalq} + DARS_Score + RRS_Score + Age + C(Sex) + avg_condition_order"
            m = smf.ols(formula, data=dat).fit()
            ci = m.conf_int().loc[finalq]
            rows.append({"FinalQ": finalq, "FinalQ_text": FINALQ_LABELS[finalq], "Outcome": outcome, "N": int(m.nobs),
                         "b": m.params[finalq], "SE": m.bse[finalq], "CI_low": ci[0], "CI_high": ci[1],
                         "p_raw": m.pvalues[finalq], "Adj_R2": m.rsquared_adj, "Formula": formula})
    results = pd.DataFrame(rows)
    results["p_FDR"] = multipletests(results["p_raw"], method="fdr_bh")[1]
    results["stars_FDR"] = results["p_FDR"].apply(p_to_stars)

    nrows, ncols = len(FINALQS), len(OUTCOMES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1*ncols, 2.65*nrows), constrained_layout=False)
    for r, finalq in enumerate(FINALQS):
        for c, outcome in enumerate(OUTCOMES):
            ax = axes[r, c]
            row = results[(results["FinalQ"].eq(finalq)) & (results["Outcome"].eq(outcome))].iloc[0]
            dat = DF.dropna(subset=[outcome, finalq, "DARS_Score", "RRS_Score", "Age", "Sex", "avg_condition_order"])
            y_res = smf.ols(f"{outcome} ~ DARS_Score + RRS_Score + Age + C(Sex) + avg_condition_order", data=dat).fit().resid
            x_res = smf.ols(f"{finalq} ~ DARS_Score + RRS_Score + Age + C(Sex) + avg_condition_order", data=dat).fit().resid
            ax.scatter(x_res, y_res, s=12, alpha=0.35)
            lr = stats.linregress(x_res, y_res)
            xs = np.linspace(np.nanmin(x_res), np.nanmax(x_res), 50)
            ax.plot(xs, lr.intercept + lr.slope * xs, lw=1.4)
            ax.axhline(0, linestyle="--", linewidth=0.8)
            ax.axvline(0, linestyle="--", linewidth=0.8)
            ax.set_title(PRETTY_OUTCOME.get(outcome, outcome), fontweight="bold")
            ax.set_xlabel(f"{finalq} residualized")
            ax.set_ylabel("Outcome residualized")
            ax.text(0.03, 0.97, f"b = {row.b:.4f}\n95% CI [{row.CI_low:.3f}, {row.CI_high:.3f}]\np = {row.p_raw:.3g}\nFDR {row.stars_FDR or 'ns'}", transform=ax.transAxes, va="top", fontsize=7.5, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d9d9d9", alpha=0.9))
            beautify_axes(ax)

    for row_idx, finalq in enumerate(FINALQS):
        fig.text(0.005, 1 - (row_idx + 0.5) / nrows, finalq, va="center", ha="left", fontsize=12, fontweight="bold", color="#4A2C6D")
    legend_text = (
        "FinalQ item wording\n"
        "FinalQ_1 – How aware are you of how your actions impact others?\n"
        "FinalQ_2 – Are you able to adjust your behavior after receiving feedback?\n"
        "FinalQ_3 – Do you unfairly blame yourself for things that happen to you?\n"
        "FinalQ_4 – Do you unfairly blame others/external circumstances for things that happen to you?"
    )
    fig.text(0.5, 0.01, legend_text, ha="center", va="bottom", fontsize=10, bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#d9d9d9", alpha=0.95))
    fig.suptitle("Associations between FinalQ items and task-derived measures\ncontrolling for DARS, RRS, age, sex, and condition order", y=0.995, fontsize=14, fontweight="bold")
    plt.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.12, wspace=0.30, hspace=0.45)
    plt.close()
    print("\n=== FinalQ item-level models ===")
    print(results[["FinalQ", "Outcome", "N", "b", "SE", "CI_low", "CI_high", "p_raw", "p_FDR", "stars_FDR", "Adj_R2"]].to_string(index=False))
    return results

# =============================================================================
# FIGURE 6: TASK SCHEMATIC
# =============================================================================


def figure6_task_schematic(save=True):
    """Simple generated schematic for the Efficacy-Tracking Task Structure."""
    steps = [
        ("1", "Fixation / cue", "Blue circle\n1-3.5 s"),
        ("2", "Number sequence", "Type target\nas fast as possible"),
        ("3", "Estimate feedback", "Predict points\nfrom performance"),
        ("4", "Point feedback", "Performance-based\nor random"),
        ("5", "Efficacy rating", "Every 3 trials:\nrate current efficacy"),
    ]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, (num, title, body) in enumerate(steps):
        x = xs[i]
        rect = plt.Rectangle((x - 0.08, 0.35), 0.16, 0.36, fill=False, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x, 0.66, f"{num}. {title}", ha="center", va="center", fontweight="bold", fontsize=9)
        ax.text(x, 0.49, body, ha="center", va="center", fontsize=8)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i+1] - 0.09, 0.53), xytext=(x + 0.09, 0.53), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.5, 0.16, "Efficacy rate drifts over time; participants infer whether feedback reflects performance or randomness.",
            ha="center", va="center", fontsize=9)
    ax.set_title("Figure 6. Efficacy-Tracking Task Structure", fontweight="bold")
    plt.close()
    return fig

# =============================================================================
# SIMULATIONS / FIGURE 7 / SUPPLEMENTAL FIGURE 2
# =============================================================================

def datageneration():
    """Restored/fixed data-generation function used for model-behavior checks."""
    percent = [
        0.5, 0.48, 0.46, 0.44, 0.42, 0.4, 0.38, 0.36, 0.34, 0.32,
        0.3, 0.28, 0.26, 0.24, 0.22, 0.2, 0.18, 0.16, 0.18,
        0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.28, 0.26, 0.24, 0.22,
        0.2, 0.18, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28,
        0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48,
        0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68,
        0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.82,
        0.8, 0.78, 0.76, 0.74, 0.72, 0.7, 0.72, 0.74, 0.76, 0.78,
        0.8, 0.82, 0.84, 0.82, 0.8, 0.78, 0.76, 0.74, 0.72,
        0.7, 0.68, 0.66, 0.64, 0.62, 0.6, 0.58, 0.56, 0.54, 0.52
    ]
    n_samples = len(percent)
    feedback_random = np.round(np.random.uniform(low=-6, high=6, size=n_samples)).astype(int)

    feedback_performance = []
    while len(feedback_performance) < n_samples:
        sample = int(np.round(np.random.normal(loc=4, scale=4)))
        if -6 <= sample <= 6:
            feedback_performance.append(sample)
    feedback_performance = np.asarray(feedback_performance, dtype=int)

    feedback_noise = np.rint(np.clip(np.random.normal(loc=-0.5, scale=1, size=n_samples), -6, 6)).astype(int)
    trial_rand = np.random.uniform(size=n_samples) >= np.asarray(percent)
    trial_perf = ~trial_rand
    feedback_actual = []
    r_i = p_i = 0
    for is_perf in trial_perf:
        if is_perf:
            feedback_actual.append(feedback_performance[trial_perf][p_i]); p_i += 1
        else:
            feedback_actual.append(feedback_random[trial_rand][r_i]); r_i += 1

    feedback_performance_reported = feedback_performance + feedback_noise

    trialtype = np.zeros(n_samples, dtype=int)
    spacing = n_samples // 32
    for i in range(32):
        idx = i * spacing + 2
        if idx < n_samples:
            trialtype[idx] = 1

    return pd.DataFrame({
        "percent": np.asarray(percent, dtype=float),
        "feedback_random": feedback_random,
        "feedback_performance": feedback_performance,
        "feedback_noise": feedback_noise,
        "feedback_performance_reported": feedback_performance_reported,
        "TrialTypeRand": trial_rand,
        "TrialTypePerf": trial_perf,
        "feedback_actual": np.asarray(feedback_actual, dtype=int),
        "trialtype": trialtype,
    })


def figure7_model_behavior(save=True):
    df = datageneration()
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = np.arange(len(df))
    ax.plot(x, df["percent"], label="Ground-truth efficacy rate")
    ax.set_title("A  Task-long efficacy rates", loc="left", fontweight="bold")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Efficacy rate")
    ax.set_ylim(0, 1)
    beautify_axes(ax)
    ax.legend(frameon=False)
    plt.close()
    return df


def supplemental_noise_landscape(noise_sds=(0.5, 1.5, 3.0, 6.0), save=True):
    """Grid-search illustration for Supplemental Figure 2.

    This is intentionally compact and interpretable rather than exhaustive. Increase grid_n for a denser figure.
    """
    base = datageneration()
    grid_n = 9
    alpha_grid = np.linspace(0.05, 0.95, grid_n)
    cutoff_grid = np.linspace(0.5, 0.95, 5)
    intercept_grid = np.linspace(0.05, 0.95, 5)
    fig, axes = plt.subplots(1, len(noise_sds), figsize=(4.2 * len(noise_sds), 3.6), constrained_layout=True)
    if len(noise_sds) == 1:
        axes = [axes]

    rows = []
    for ax, sd in zip(axes, noise_sds):
        sim = base.copy()
        sim["feedback_performance_reported"] = sim["feedback_performance"] + np.rint(np.clip(np.random.normal(0, sd, size=len(sim)), -6, 6)).astype(int)
        accum = sim.loc[sim["trialtype"].eq(1), "percent"].to_numpy(dtype=float)
        f_actual = sim["feedback_actual"]
        f_perf = sim["feedback_performance_reported"]
        trialtype = sim["trialtype"].to_numpy()
        vals = []
        best = None
        for ap, an, cut, inter in product(alpha_grid, alpha_grid, cutoff_grid, intercept_grid):
            sse = twoLR_Cutoff_intercept_b_drifting_trial([ap, an, cut, inter], f_actual, f_perf, accum, trialtype, len(sim))
            vals.append((ap, an, cut, inter, sse))
            if best is None or sse < best[-1]:
                best = vals[-1]
        vdf = pd.DataFrame(vals, columns=["alpha_p", "alpha_n", "cutoff", "intercept", "SSE"])
        sc = ax.scatter(vdf["alpha_p"], vdf["alpha_n"], c=np.log(vdf["SSE"] + 1e-8), s=18)
        ax.set_title(f"Noise SD = {sd}", fontweight="bold")
        ax.set_xlabel("α_p")
        ax.set_ylabel("α_n")
        ax.text(0.03, 0.97, f"min log(SSE)={np.log(best[-1]+1e-8):.2f}\nαp={best[0]:.2f}, αn={best[1]:.2f}\ncut={best[2]:.2f}, int={best[3]:.2f}", transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        beautify_axes(ax)
        rows.append({"noise_sd": sd, "best_alpha_p": best[0], "best_alpha_n": best[1], "best_cutoff": best[2], "best_intercept": best[3], "min_SSE": best[4]})
    fig.colorbar(sc, ax=axes, label="log(SSE)")
    plt.close()
    return pd.DataFrame(rows)

# =============================================================================
# SENSITIVITY / DIAGNOSTICS USED FOR CHECKING MODELS
# =============================================================================

def regression_diagnostics(df: pd.DataFrame, formula: str, focal_term: str | None = None):
    dat = df.dropna(subset=[x.strip().split("(")[-1].rstrip(")") for x in formula.replace("~", "+").split("+") if x.strip() and ":" not in x and "C(" not in x]).copy()
    model = smf.ols(formula, data=dat).fit()
    infl = OLSInfluence(model)
    p = int(model.df_model) + 1
    n = int(model.nobs)
    df_diag = dat.copy()
    df_diag["hat"] = infl.hat_matrix_diag
    df_diag["studentized_resid"] = infl.resid_studentized_external
    df_diag["cooks_d"] = infl.cooks_distance[0]
    df_diag["high_leverage"] = df_diag["hat"] > (2 * p / n)
    df_diag["large_resid"] = df_diag["studentized_resid"].abs() > 3
    df_diag["high_cook"] = df_diag["cooks_d"] > (4 / n)
    df_diag["any_influential"] = df_diag[["high_leverage", "large_resid", "high_cook"]].any(axis=1)
    model_clean = smf.ols(formula, data=df_diag.loc[~df_diag["any_influential"]]).fit()
    model_hc3 = model.get_robustcov_results(cov_type="HC3")

    print("\n=== Regression diagnostics ===")
    print("Formula:", formula)
    print("Flagged influential cases:", int(df_diag["any_influential"].sum()))
    if focal_term is not None:
        comp = pd.DataFrame({
            "Model": ["Original", "Excluding influential cases"],
            "N": [int(model.nobs), int(model_clean.nobs)],
            "b": [model.params.get(focal_term, np.nan), model_clean.params.get(focal_term, np.nan)],
            "SE": [model.bse.get(focal_term, np.nan), model_clean.bse.get(focal_term, np.nan)],
            "p": [model.pvalues.get(focal_term, np.nan), model_clean.pvalues.get(focal_term, np.nan)],
            "Adj_R2": [model.rsquared_adj, model_clean.rsquared_adj],
        })
        print(comp.to_string(index=False))
    return model, model_clean, model_hc3, df_diag


def export_embedded_manuscript_figures(
    docx_path: Path = MANUSCRIPT_DOCX_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Export the exact raster figures embedded in the manuscript.

    The manuscript figures were manually scaled/cropped in Word. Re-rendering
    the plotting code can produce visually similar but not identical images.
    This exporter preserves the exact embedded rasters so the files in
    `analysis_outputs` match the current manuscript figures byte-for-byte.
    """
    if not docx_path.exists():
        print(f"Skipping manuscript figure export; missing DOCX: {docx_path}")
        return

    media_to_output = {
        "word/media/image1.png": "figure1_task_schematic.png",
        "word/media/image2.png": "figure2_task_performance_by_group.png",
        "word/media/image3.png": "figure3_condition_difference_regressions.png",
        "word/media/image4.png": "figure4_model_fit.png",
        "word/media/image5.png": "figure5_learning_rate_associations.png",
        "word/media/image6.png": "figure6_finalq_latent_factor_associations.png",
        "word/media/image7.png": "figure7_model_behavior_and_recovery.png",
        "word/media/image8.png": "supplemental_figure1_finalq_group_means.png",
        "word/media/image9.png": "supplemental_figure2_noise_landscape.png",
    }

    manifest_rows = []
    with ZipFile(docx_path) as zf:
        for media_name, output_name in media_to_output.items():
            if media_name not in zf.namelist():
                continue
            out_path = output_dir / output_name
            with zf.open(media_name) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            manifest_rows.append({"manuscript_media": media_name, "output_file": output_name})

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(output_dir / "manuscript_figure_export_manifest.csv", index=False)
        print("\nExported exact embedded manuscript figures to:", output_dir)

# =============================================================================
# MAIN
# =============================================================================

def main():
    raw_data = load_raw_data()
    compute_performance_training_noise(raw_data)
    compute_phase2_accuracy(raw_data)

    clean_data = build_clean_task_data(raw_data)
    report_dataframe_level(clean_data, "clean_data task table")

    gooddata = fit_model_all_participants(clean_data, n_starts=N_FIT_STARTS)
    assert_one_row_per_participant(gooddata, "gooddata model-parameter table")

    metrics, order_df = compute_behavioral_metrics(clean_data, gooddata)
    assert_one_row_per_participant(metrics, "participant_task_metrics")

    fit_df = build_fit_df(clean_data, gooddata)
    report_dataframe_level(fit_df, "fit_df model-fit table")

    questionnaires, demographics = load_questionnaires_and_demographics()
    finalq = extract_finalq_from_raw(raw_data)
    final_df = make_final_df(questionnaires, demographics, metrics, finalq=finalq)

    # Run participant-level learning-rate group comparisons here, after final_df exists.
    learning_rate_ancova, learning_rate_pairwise = compare_learning_rates_by_group(final_df)

    # Save core tables so every downstream figure/result can be reproduced.
    clean_data.to_csv(OUTPUT_DIR / "clean_data.csv", index=False)
    gooddata.to_csv(OUTPUT_DIR / "model_parameters_gooddata.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "participant_task_metrics.csv", index=False)
    fit_df.to_csv(OUTPUT_DIR / "model_fit_triallevel.csv", index=False)
    finalq.to_csv(OUTPUT_DIR / "finalq_from_raw_data.csv", index=False)
    final_df.to_csv(OUTPUT_DIR / "final_df_one_row_per_participant.csv", index=False)

    figure2_ancova = figure1_task_performance_by_group(final_df)
    figure2_ancova.to_csv(OUTPUT_DIR / "figure2_task_performance_ANCOVA_summary.csv", index=False)

    condition_difference_model = figure2_condition_difference_regressions(final_df)
    condition_difference_model.summary2().tables[1].to_csv(
        OUTPUT_DIR / "figure3_condition_difference_regression_coefficients.csv"
    )

    rmse_summary = figure3_model_fit(fit_df, final_df)
    rmse_summary.to_csv(OUTPUT_DIR / "figure4_model_fit_RMSE_summary.csv")

    learning_rate_dimensional = figure4_learning_rate_associations(final_df)
    learning_rate_dimensional.to_csv(OUTPUT_DIR / "figure5_learning_rate_dimensional_models.csv", index=False)

    if HAS_FACTOR_ANALYZER:
        _, finalq_factor_results, finalq_loadings = run_finalq_factor_analysis(final_df)
        finalq_factor_results.to_csv(OUTPUT_DIR / "figure6_finalq_factor_associations.csv", index=False)
        finalq_loadings.to_csv(OUTPUT_DIR / "figure6_finalq_factor_loadings.csv")
    else:
        print("Skipping Figure 6 factor analysis because factor_analyzer is not installed.")

    finalq_omnibus, finalq_posthoc = supplemental_finalq_group_means(final_df)
    finalq_omnibus.to_csv(OUTPUT_DIR / "supplemental_finalq_group_omnibus.csv", index=False)
    finalq_posthoc.to_csv(OUTPUT_DIR / "supplemental_finalq_group_posthoc.csv", index=False)

    finalq_item_results = finalq_item_level_models(final_df)
    finalq_item_results.to_csv(OUTPUT_DIR / "supplemental_finalq_item_level_models.csv", index=False)
    figure6_task_schematic()
    figure7_model_behavior()
    supplemental_noise_landscape()
    export_embedded_manuscript_figures()

    print("\nAll requested cleaned outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
