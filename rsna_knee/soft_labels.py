"""Train-only soft targets for the 12 findings.

Reports exist on train studies and not on the hidden test. This module never
reads a Report column. UNK is silence, not a negative: the target stays the
steven v4 soft score, and the sample weight is the labeler's confidence.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from rsna_knee.metrics import macro_roc_auc, per_finding_roc_auc
from rsna_knee.schema import FINDINGS, ID_COLUMN
from rsna_knee.v4_split import expert_gold_ids

RARE_FINDINGS: tuple[str, ...] = ("MCL", "Baker's", "Fracture", "Synovitis")


def _as_id_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if ID_COLUMN not in frame.columns:
        raise ValueError("table missing StudyInstanceUID")
    out = frame.copy()
    out[ID_COLUMN] = out[ID_COLUMN].astype(str)
    return out.drop_duplicates(ID_COLUMN)


def rare_unk_rates(pilkwang: pd.DataFrame) -> dict[str, float]:
    """Fraction of rows whose pilkwang verdict is UNK, for the rare findings."""
    frame = _as_id_frame(pilkwang)
    rates: dict[str, float] = {}
    for name in RARE_FINDINGS:
        column = f"{name}__verdict"
        if column not in frame.columns:
            raise ValueError(f"pilkwang table missing {column}")
        verdict = frame[column].astype(str)
        rates[name] = float((verdict == "UNK").mean()) if len(frame) else float("nan")
    return rates


def grade_soft_targets(
    v4: pd.DataFrame,
    pilkwang: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Soft scores plus confidence weights. UNK is never written as 0.

    Targets are the steven v4 scores, unchanged and not snapped to 0 or 1.
    Weights are pilkwang ``__conf`` clipped to [0, 1]. A row with no pilkwang
    verdict keeps weight 1. Gold rows are still present; drop them with
    ``non_gold_train_ids`` before a loss.
    """
    left = _as_id_frame(v4)
    missing = [name for name in FINDINGS if name not in left.columns]
    if missing:
        raise ValueError(f"v4 table missing findings {missing}")
    right = _as_id_frame(pilkwang)
    conf_cols = [f"{name}__conf" for name in FINDINGS]
    verdict_cols = [f"{name}__verdict" for name in FINDINGS]
    absent = [column for column in conf_cols + verdict_cols if column not in right.columns]
    if absent:
        raise ValueError(f"pilkwang table missing {absent}")

    merged = left[[ID_COLUMN, *FINDINGS]].merge(
        right[[ID_COLUMN, *conf_cols, *verdict_cols]],
        on=ID_COLUMN,
        how="left",
    )
    targets = merged[[ID_COLUMN, *FINDINGS]].copy()
    weights = pd.DataFrame({ID_COLUMN: merged[ID_COLUMN]})
    for name in FINDINGS:
        score = pd.to_numeric(merged[name], errors="coerce")
        if score.isna().any():
            raise ValueError(f"v4 score for {name} has missing values")
        targets[name] = score.to_numpy(dtype=np.float64)
        conf = pd.to_numeric(merged[f"{name}__conf"], errors="coerce").clip(0.0, 1.0)
        verdict = merged[f"{name}__verdict"]
        weight = conf.to_numpy(dtype=np.float64)
        missing_verdict = verdict.isna().to_numpy()
        weight = np.where(missing_verdict, 1.0, np.where(np.isfinite(weight), weight, 0.0))
        weights[name] = weight
    return targets, weights


def confidence_graded_targets(
    v4: pd.DataFrame,
    pilkwang: pd.DataFrame,
) -> pd.DataFrame:
    """Grade a known mention into the v4 score. UNK stays the v4 score.

    A non-UNK row uses ``conf * pilkwang + (1 - conf) * v4``. Silence is not
    written as 0. Gold labels are not an input.
    """
    left = _as_id_frame(v4)
    right = _as_id_frame(pilkwang)
    conf_cols = [f"{name}__conf" for name in FINDINGS]
    verdict_cols = [f"{name}__verdict" for name in FINDINGS]
    missing = [name for name in FINDINGS if name not in left.columns or name not in right.columns]
    if missing:
        raise ValueError(f"graded targets missing findings {missing}")
    absent = [column for column in conf_cols + verdict_cols if column not in right.columns]
    if absent:
        raise ValueError(f"pilkwang table missing {absent}")
    merged = left[[ID_COLUMN, *FINDINGS]].merge(
        right[[ID_COLUMN, *FINDINGS, *conf_cols, *verdict_cols]],
        on=ID_COLUMN,
        how="left",
        suffixes=("_v4", "_pw"),
    )
    out = pd.DataFrame({ID_COLUMN: merged[ID_COLUMN]})
    for name in FINDINGS:
        v4_score = pd.to_numeric(merged[f"{name}_v4"], errors="coerce").to_numpy(dtype=np.float64)
        pw_score = pd.to_numeric(merged[f"{name}_pw"], errors="coerce").to_numpy(dtype=np.float64)
        conf = pd.to_numeric(merged[f"{name}__conf"], errors="coerce").clip(0.0, 1.0).to_numpy(dtype=np.float64)
        verdict = merged[f"{name}__verdict"]
        known = verdict.notna().to_numpy() & ~verdict.astype(str).eq("UNK").to_numpy()
        known &= np.isfinite(conf) & np.isfinite(pw_score) & np.isfinite(v4_score)
        graded = v4_score.copy()
        graded[known] = conf[known] * pw_score[known] + (1.0 - conf[known]) * v4_score[known]
        out[name] = graded
    return out


def harden_unk_to_zero(pilkwang: pd.DataFrame) -> pd.DataFrame:
    """The failure mode: map UNK verdicts to target 0. Do not train on this."""
    frame = _as_id_frame(pilkwang)
    out = frame[[ID_COLUMN]].copy()
    for name in FINDINGS:
        score = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        unk = frame[f"{name}__verdict"].astype(str).eq("UNK").to_numpy()
        score = score.copy()
        score[unk] = 0.0
        out[name] = score
    return out


def non_gold_train_ids(ids: pd.Series, gold_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Train IDs with every expert gold UID removed."""
    gold = {str(uid) for uid in gold_ids}
    kept = tuple(str(uid) for uid in ids.tolist() if str(uid) not in gold)
    if set(kept) & gold:
        raise ValueError("gold UID leaked into soft-label train ids")
    return kept


def unk_loss_share(weights: pd.DataFrame, pilkwang: pd.DataFrame) -> dict[str, float]:
    """Share of confidence-weight mass that sits on UNK rows."""
    frame = _as_id_frame(pilkwang)
    merged = weights.merge(
        frame[[ID_COLUMN, *[f"{name}__verdict" for name in RARE_FINDINGS]]],
        on=ID_COLUMN,
        how="inner",
    )
    shares: dict[str, float] = {}
    for name in RARE_FINDINGS:
        weight = pd.to_numeric(merged[name], errors="coerce").to_numpy(dtype=np.float64)
        unk = merged[f"{name}__verdict"].astype(str).eq("UNK").to_numpy()
        total = float(np.nansum(weight))
        shares[name] = float(np.nansum(weight[unk]) / total) if total > 0.0 else float("nan")
    return shares


def _gold_scores(
    train: pd.DataFrame,
    scores: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    gold = expert_gold_ids(train)
    truth = _as_id_frame(train)
    truth = truth[truth[ID_COLUMN].isin(gold)].set_index(ID_COLUMN)
    pred = _as_id_frame(scores).set_index(ID_COLUMN)
    common = [uid for uid in gold if uid in pred.index and uid in truth.index]
    if len(common) < 2:
        raise ValueError("gold overlap with scores is below 2 rows")
    y = truth.loc[common, list(FINDINGS)].to_numpy(dtype=np.float64)
    s = pred.loc[common, list(FINDINGS)].to_numpy(dtype=np.float64)
    return y, s


def rare_gold_aucs(train: pd.DataFrame, scores: pd.DataFrame) -> dict[str, float]:
    """Expert-gold ROC AUC for the rare findings. Gold is val-only."""
    y, s = _gold_scores(train, scores)
    aucs = per_finding_roc_auc(y, s)
    index = {name: i for i, name in enumerate(FINDINGS)}
    return {name: float(aucs[index[name]]) for name in RARE_FINDINGS}


def bootstrap_macro_delta(
    y_true: np.ndarray,
    base_scores: np.ndarray,
    new_scores: np.ndarray,
    *,
    n_draws: int = 400,
    seed: int = 2026,
) -> dict[str, float]:
    """Paired row bootstrap of macro AUC(new) minus macro AUC(base).

    Seed 2026 and 400 draws match the ConvNeXt private-gate boots. The point
    is the full-sample delta. ``lo`` and ``hi`` are the 2.5 and 97.5 percentiles.
    """
    y = np.asarray(y_true, dtype=np.float64)
    base = np.asarray(base_scores, dtype=np.float64)
    new = np.asarray(new_scores, dtype=np.float64)
    if y.ndim != 2 or y.shape != base.shape or y.shape != new.shape:
        raise ValueError("bootstrap arrays must be 2D and the same shape")
    if y.shape[0] < 2 or n_draws < 1:
        raise ValueError("bootstrap needs at least 2 rows and 1 draw")
    point = float(macro_roc_auc(y, new) - macro_roc_auc(y, base))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_draws), dtype=np.float64)
    n = int(y.shape[0])
    for i in range(int(n_draws)):
        take = rng.integers(0, n, size=n)
        draws[i] = float(macro_roc_auc(y[take], new[take]) - macro_roc_auc(y[take], base[take]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "n": float(n),
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_draws": float(n_draws),
    }


def graft_label_clears(delta: Mapping[str, float]) -> bool:
    """True only when the grafted-label macro CI sits strictly above 0."""
    point = float(delta["point"])
    lo = float(delta["lo"])
    return bool(np.isfinite(point) and np.isfinite(lo) and point > 0.0 and lo > 0.0)


def graft_versus_v4_gold_delta(
    train: pd.DataFrame,
    v4: pd.DataFrame,
    pilkwang: pd.DataFrame,
    *,
    n_draws: int = 400,
    seed: int = 2026,
) -> dict[str, float]:
    """Gold-row macro delta of the Fracture graft against steven v4.

    Gold labels are the metric, not a training input. Rows missing from
    pilkwang keep the v4 Fracture score.
    """
    grafted = graft_finding_scores(v4, pilkwang, FRACTURE_GRAFT_FINDINGS)
    y_base, s_base = _gold_scores(train, v4)
    y_new, s_new = _gold_scores(train, grafted)
    if y_base.shape != y_new.shape or not np.array_equal(y_base, y_new):
        raise ValueError("graft and v4 gold rows are not the same expert studies")
    return bootstrap_macro_delta(y_base, s_base, s_new, n_draws=n_draws, seed=seed)


def gold_finding_aucs(train: pd.DataFrame, scores: pd.DataFrame) -> dict[str, float]:
    """Expert-gold ROC AUC for all 12 findings, plus macro and n_gold.

    Gold is val-only. ``n_gold`` is the overlap count, not an AUC.
    """
    y, s = _gold_scores(train, scores)
    aucs = per_finding_roc_auc(y, s)
    out = {name: float(aucs[i]) for i, name in enumerate(FINDINGS)}
    out["macro"] = float(np.nanmean(aucs))
    out["n_gold"] = float(y.shape[0])
    return out


def rank_average_score_tables(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Equal rank-average of soft-score tables. Train target only, not a test overlay.

    Studies must sit in every table. A missing finding or a missing score raises.
    UNK is not rewritten: each table's own score is ranked as stored.
    """
    if len(frames) < 2:
        raise ValueError("rank average needs at least two score tables")
    indexed: list[pd.DataFrame] = []
    for frame in frames:
        prepared = _as_id_frame(frame)
        missing = [name for name in FINDINGS if name not in prepared.columns]
        if missing:
            raise ValueError(f"score table missing findings {missing}")
        scores = prepared.set_index(ID_COLUMN)[list(FINDINGS)]
        scores = scores.apply(pd.to_numeric, errors="coerce")
        if scores.isna().any().any():
            raise ValueError("score table has a missing finding value")
        indexed.append(scores)
    common = [uid for uid in indexed[0].index if all(uid in frame.index for frame in indexed[1:])]
    if len(common) < 2:
        raise ValueError("score tables share fewer than 2 studies")
    total = None
    for frame in indexed:
        ranks = frame.loc[common, list(FINDINGS)].rank(method="average", pct=True)
        total = ranks if total is None else total.add(ranks)
    assert total is not None
    averaged = (total / float(len(indexed))).reset_index()
    return averaged[[ID_COLUMN, *FINDINGS]]


# Pilkwang Fracture gold AUC beat steven v4 by +0.0801 on the 57 shared expert
# rows (measured 2026-10-08). No other pilkwang column beat v4 by more than
# +0.0109. Replacing all 12 columns dropped macro by 0.0232. Graft Fracture only.
FRACTURE_GRAFT_FINDINGS: tuple[str, ...] = ("Fracture",)


def graft_finding_scores(
    base: pd.DataFrame,
    donor: pd.DataFrame,
    findings: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Copy donor scores onto base for named findings. Missing donor rows keep base.

    Train target only. Does not read Report, does not take gold labels, and does
    not rewrite UNK to 0.
    """
    chosen = tuple(findings)
    if not chosen:
        raise ValueError("graft needs at least one finding")
    unknown = [name for name in chosen if name not in FINDINGS]
    if unknown:
        raise ValueError(f"graft findings not in the schema: {unknown}")
    left = _as_id_frame(base)
    right = _as_id_frame(donor)
    missing = [name for name in FINDINGS if name not in left.columns]
    missing += [name for name in chosen if name not in right.columns]
    if missing:
        raise ValueError(f"graft tables missing {missing}")
    base_scores = left[[ID_COLUMN, *FINDINGS]].copy()
    for name in FINDINGS:
        base_scores[name] = pd.to_numeric(base_scores[name], errors="coerce")
        if base_scores[name].isna().any():
            raise ValueError(f"base score for {name} has missing values")
    donor_scores = right[[ID_COLUMN, *chosen]].copy()
    for name in chosen:
        donor_scores[name] = pd.to_numeric(donor_scores[name], errors="coerce")
    merged = base_scores.merge(donor_scores, on=ID_COLUMN, how="left", suffixes=("", "_donor"))
    out = base_scores.copy()
    for name in chosen:
        donor_column = f"{name}_donor" if f"{name}_donor" in merged.columns else name
        donor_values = merged[donor_column]
        out[name] = merged[name].where(donor_values.isna(), donor_values).to_numpy(dtype=np.float64)
    return out[[ID_COLUMN, *FINDINGS]]


def fracture_graft_train_frame(
    v4: pd.DataFrame,
    pilkwang: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Non-gold train targets: steven v4 with the pilkwang Fracture column copied on.

    Expert gold rows are removed. Report is not read. This frame is the loss
    target for the next frozen head, not a test-time overlay.
    """
    grafted = graft_finding_scores(v4, pilkwang, FRACTURE_GRAFT_FINDINGS)
    gold = expert_gold_ids(train)
    keep = set(non_gold_train_ids(grafted[ID_COLUMN], gold))
    out = grafted[grafted[ID_COLUMN].astype(str).isin(keep)].copy()
    leaked = set(out[ID_COLUMN].astype(str)) & set(gold)
    if leaked:
        raise ValueError("gold UID leaked into the fracture graft train frame")
    if "Report" in out.columns:
        raise ValueError("fracture graft train frame must not carry Report")
    return out.reset_index(drop=True)


def fracture_graft_train_bundle(
    v4: pd.DataFrame,
    pilkwang: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Non-gold targets and confidence weights for the grafted Fracture head.

    Targets are ``fracture_graft_train_frame``. Weights are pilkwang ``__conf``.
    Gold UIDs are absent from both frames. Report is not read.
    """
    targets = fracture_graft_train_frame(v4, pilkwang, train)
    _soft, weights = grade_soft_targets(v4, pilkwang)
    ordered = (
        weights.drop_duplicates(ID_COLUMN)
        .set_index(ID_COLUMN)
        .reindex(targets[ID_COLUMN].tolist())
        .reset_index()
    )
    if list(ordered[ID_COLUMN].astype(str)) != list(targets[ID_COLUMN].astype(str)):
        raise ValueError("graft train bundle id order drifted")
    if ordered[list(FINDINGS)].isna().any().any():
        raise ValueError("graft train bundle is missing a weight")
    gold = set(expert_gold_ids(train))
    if set(ordered[ID_COLUMN].astype(str)) & gold:
        raise ValueError("gold UID leaked into the fracture graft weights")
    return targets, ordered[[ID_COLUMN, *FINDINGS]]


def graft_fit_matrices(
    v4: pd.DataFrame,
    pilkwang: pd.DataFrame,
    train: pd.DataFrame,
    fit_ids: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Loss matrices for a frozen head. Gold IDs raise. Report is not read.

    ``y`` is the Fracture graft. ``w`` is the pilkwang confidence. Both rows
    follow ``fit_ids``.
    """
    ids = [str(uid) for uid in fit_ids]
    if len(ids) < 2:
        raise ValueError("graft fit needs at least 2 studies")
    if len(set(ids)) != len(ids):
        raise ValueError("graft fit ids are not unique")
    gold = set(expert_gold_ids(train))
    leaked = set(ids) & gold
    if leaked:
        raise ValueError("gold UID in graft fit ids")
    targets, weights = fracture_graft_train_bundle(v4, pilkwang, train)
    y_idx = targets.set_index(ID_COLUMN)
    w_idx = weights.set_index(ID_COLUMN)
    missing = [uid for uid in ids if uid not in y_idx.index or uid not in w_idx.index]
    if missing:
        raise ValueError(f"graft fit missing {len(missing)} studies")
    y = y_idx.loc[ids, list(FINDINGS)].to_numpy(dtype=np.float64)
    w = w_idx.loc[ids, list(FINDINGS)].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError("graft fit has a non-finite value")
    return y, w


def audit_rare_labels(
    train: pd.DataFrame,
    v4: pd.DataFrame,
    pilkwang: pd.DataFrame,
) -> Mapping[str, object]:
    """Silence rates, confidence-weight shares, and gold AUCs. No Report read."""
    targets, weights = grade_soft_targets(v4, pilkwang)
    hardened = harden_unk_to_zero(pilkwang)
    gold = expert_gold_ids(train)
    train_ids = non_gold_train_ids(targets[ID_COLUMN], gold)
    return {
        "n_v4": int(len(_as_id_frame(v4))),
        "n_pilkwang": int(len(_as_id_frame(pilkwang))),
        "n_gold": int(len(gold)),
        "n_gold_in_v4": int(_gold_scores(train, targets)[0].shape[0]),
        "n_gold_in_pilkwang": int(_gold_scores(train, pilkwang)[0].shape[0]),
        "n_train": int(len(train_ids)),
        "unk_rate": rare_unk_rates(pilkwang),
        "unk_loss_share": unk_loss_share(weights, pilkwang),
        "gold_auc_v4": rare_gold_aucs(train, targets),
        "gold_auc_pilkwang": rare_gold_aucs(train, pilkwang),
        "gold_auc_hardened": rare_gold_aucs(train, hardened),
    }
