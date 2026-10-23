"""Private RadImageNet-DenseNet121 head. Do not Submit.

Fit on non-gold LLM-soft rows except the mill-870 holdout. Gold macro above
0.94 aborts before a holdout file is written.
"""
from __future__ import annotations

ID_COLUMN = "StudyInstanceUID"
FINDINGS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)

"""Macro-averaged ROC AUC over the 12 findings (host score)."""


import numpy as np
from sklearn.metrics import roc_auc_score


def per_finding_roc_auc(y_true, y_score) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_score must be 2D arrays of the same shape")
    aucs = np.empty(y_true.shape[1], dtype=np.float64)
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            aucs[i] = np.nan
        else:
            aucs[i] = float(roc_auc_score(y_true[:, i], y_score[:, i]))
    return aucs


def macro_roc_auc(y_true, y_score) -> float:
    """Mean of per-finding ROC AUC — the competition metric.

    Findings that are single-class in a split are skipped (nanmean), matching
    the public 2.5D ResNet34 baseline's compute_macro_auc guard.
    """
    aucs = per_finding_roc_auc(y_true, y_score)
    if np.isnan(aucs).all():
        return 0.5
    return float(np.nanmean(aucs))

"""Steven v4 train split: expert gold 58 is val-only and never in the loss.

Train targets come from stevenleehans `llm_labels_v4_blend.csv`. Val labels come
from the 58 expert rows in `train.csv`. The one gold UID missing from some LLM
tables still stays val-only.
"""


from dataclasses import dataclass

import numpy as np
import pandas as pd


GOLD_AUC_ABORT_THRESHOLD = 0.94


class GoldOverfitAbort(RuntimeError):
    """Gold-58 macro AUC is ≫ 0.94 — leak/overfit alarm. Do not overlay."""


@dataclass(frozen=True)
class StevenV4Split:
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    y_train: np.ndarray
    y_val: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train_ids)

    @property
    def n_val(self) -> int:
        return len(self.val_ids)


def expert_gold_ids(train: pd.DataFrame) -> tuple[str, ...]:
    missing = [c for c in (ID_COLUMN, *FINDINGS) if c not in train.columns]
    if missing:
        raise ValueError(f"train table missing columns {missing}")
    frame = train.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    mask = frame[list(FINDINGS)].notna().all(axis=1)
    ids = tuple(frame.loc[mask, ID_COLUMN].tolist())
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate gold StudyInstanceUID")
    return ids


def should_abort_gold_overfit(
    gold_macro: float, threshold: float = GOLD_AUC_ABORT_THRESHOLD
) -> bool:
    """True when gold macro AUC is strictly greater than 0.94 (≫ 0.94 alarm)."""
    return float(gold_macro) > float(threshold)


def abort_if_gold_overfit(
    gold_macro: float, threshold: float = GOLD_AUC_ABORT_THRESHOLD
) -> None:
    if should_abort_gold_overfit(gold_macro, threshold=threshold):
        raise GoldOverfitAbort(
            f"gold macro AUC {float(gold_macro):.4f} > {float(threshold):.2f} "
            "(leak/overfit alarm; gold 58 must stay val-only; do not overlay)"
        )


# A new frozen head must beat this mill-870 macro before any overlay.
# probe-b's measured holdout was 0.7453 and that run was already a stop.
HOLDOUT_MACRO_STOP = 0.745


def frozen_head_gate(gold_macro: float, holdout_macro: float) -> str:
    """abort, stop, or continue for a frozen head. Never an overlay by itself.

    Gold macro strictly above 0.94 aborts. A mill-870 holdout at or below
    0.745 stops. Non-finite inputs stop. Continue means the private numbers
    cleared both gates and a later overlay law still applies.
    """
    gold = float(gold_macro)
    hold = float(holdout_macro)
    if not np.isfinite(gold) or not np.isfinite(hold):
        return "stop"
    if should_abort_gold_overfit(gold):
        return "abort"
    if hold <= HOLDOUT_MACRO_STOP:
        return "stop"
    return "continue"


def make_steven_v4_split(
    train: pd.DataFrame,
    v4: pd.DataFrame,
    *,
    gold_ids: tuple[str, ...] | None = None,
) -> StevenV4Split:
    """Train = v4 rows whose UID is not gold. Val = all expert gold 58.

    Val y is the expert 0/1 matrix from train.csv, not the LLM scores.
    Train y is the steven v4 soft scores. Gold UIDs never appear in train_ids.
    """
    gold = expert_gold_ids(train) if gold_ids is None else tuple(str(x) for x in gold_ids)
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("no expert gold rows")

    tr = train.copy()
    tr[ID_COLUMN] = tr[ID_COLUMN].astype(str)
    gold_frame = tr[tr[ID_COLUMN].isin(gold_set)].drop_duplicates(ID_COLUMN)
    if len(gold_frame) != len(gold_set):
        raise ValueError("gold UIDs missing from train.csv")
    gold_frame = gold_frame.set_index(ID_COLUMN).loc[list(gold)]
    y_val = gold_frame[list(FINDINGS)].to_numpy(dtype=np.float64)

    labels = v4.copy()
    if ID_COLUMN not in labels.columns:
        raise ValueError("steven v4 table missing StudyInstanceUID")
    missing_f = [c for c in FINDINGS if c not in labels.columns]
    if missing_f:
        raise ValueError(f"steven v4 missing findings {missing_f}")
    labels[ID_COLUMN] = labels[ID_COLUMN].astype(str)
    labels = labels.drop_duplicates(ID_COLUMN)
    train_frame = labels[~labels[ID_COLUMN].isin(gold_set)].copy()
    if train_frame[ID_COLUMN].isin(gold_set).any():
        raise ValueError("gold UID leaked into steven v4 train split")
    train_ids = tuple(train_frame[ID_COLUMN].tolist())
    y_train = train_frame[list(FINDINGS)].to_numpy(dtype=np.float64)
    if set(train_ids) & gold_set:
        raise ValueError("gold 58 must never be in the train loss")
    return StevenV4Split(
        train_ids=train_ids,
        val_ids=gold,
        y_train=y_train,
        y_val=y_val,
    )


def llm_soft_holdout_indices(
    n: int, *, frac: float = 0.2, seed: int = 2026
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic fit/holdout indices over non-gold steven v4 rows.

    Gold 58 is already excluded from `n`. Holdout y is LLM-soft (not expert gold).
    """
    if n < 2:
        raise ValueError("need at least 2 non-gold rows for an LLM-soft holdout")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    n_ho = max(1, min(int(n) - 1, int(round(int(n) * float(frac)))))
    holdout = np.sort(perm[:n_ho])
    fit = np.sort(perm[n_ho:])
    if set(fit.tolist()) & set(holdout.tolist()):
        raise ValueError("LLM-soft holdout leaked into fit")
    return fit, holdout

"""Train-only soft targets for the 12 findings.

Reports exist on train studies and not on the hidden test. This module never
reads a Report column. UNK is silence, not a negative: the target stays the
steven v4 soft score, and the sample weight is the labeler's confidence.
"""


from typing import Mapping

import numpy as np
import pandas as pd


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

"""Load a Keras RadImageNet DenseNet121-notop file into torchvision.

The public file is ``sjudal/radimagenet-densenet121``
``RadImageNet-DenseNet121_notop.h5``. It is a 3-channel medical conv, not mill
DINO, CoAtNet, or RadImageNet-ResNet-50. The classifier is a new 12-way head
and is not read from the file.
"""


from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torchvision.models import densenet121

BLOCK_LAYERS = (6, 12, 24, 16)


def _dataset(group: h5py.Group, kind: str) -> np.ndarray:
    hits: list[np.ndarray] = []

    def walk(node: h5py.Group) -> None:
        for key, value in node.items():
            if isinstance(value, h5py.Dataset) and key.startswith(kind):
                hits.append(np.asarray(value))
            elif isinstance(value, h5py.Group):
                walk(value)

    walk(group)
    if len(hits) != 1:
        raise ValueError(f"{group.name} has {len(hits)} datasets for {kind}")
    return hits[0]


def _conv(kernel_hwio: np.ndarray) -> torch.Tensor:
    # Keras HWIO -> torch OIHW.
    array = np.transpose(kernel_hwio, (3, 2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(array))


def _bn(group: h5py.Group) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.from_numpy(np.ascontiguousarray(_dataset(group, "gamma"))),
        "bias": torch.from_numpy(np.ascontiguousarray(_dataset(group, "beta"))),
        "running_mean": torch.from_numpy(np.ascontiguousarray(_dataset(group, "moving_mean"))),
        "running_var": torch.from_numpy(np.ascontiguousarray(_dataset(group, "moving_variance"))),
    }


def keras_feature_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Map the notop Keras tensors onto torchvision ``features.*`` keys."""
    weights = h5py.File(path, "r")["model_weights"]
    state: dict[str, torch.Tensor] = {}
    state["features.conv0.weight"] = _conv(_dataset(weights["conv1"]["conv"], "kernel"))
    for name, tensor in _bn(weights["conv1"]["bn"]).items():
        state[f"features.norm0.{name}"] = tensor
    for name, tensor in _bn(weights["bn"]).items():
        state[f"features.norm5.{name}"] = tensor

    for block_i, n_layers in enumerate(BLOCK_LAYERS, start=1):
        keras_conv = block_i + 1
        for layer_j in range(1, n_layers + 1):
            prefix = f"features.denseblock{block_i}.denselayer{layer_j}"
            base = f"conv{keras_conv}_block{layer_j}"
            for name, tensor in _bn(weights[f"{base}_0_bn"]).items():
                state[f"{prefix}.norm1.{name}"] = tensor
            state[f"{prefix}.conv1.weight"] = _conv(_dataset(weights[f"{base}_1_conv"], "kernel"))
            for name, tensor in _bn(weights[f"{base}_1_bn"]).items():
                state[f"{prefix}.norm2.{name}"] = tensor
            state[f"{prefix}.conv2.weight"] = _conv(_dataset(weights[f"{base}_2_conv"], "kernel"))
        if block_i < 4:
            trans = f"features.transition{block_i}"
            pool = f"pool{block_i + 1}"
            for name, tensor in _bn(weights[f"{pool}_bn"]).items():
                state[f"{trans}.norm.{name}"] = tensor
            state[f"{trans}.conv.weight"] = _conv(_dataset(weights[f"{pool}_conv"], "kernel"))
    return state


def load_radimagenet_densenet121(path: str | Path) -> tuple[nn.Module, dict[str, object]]:
    """Frozen features plus a new 12-logit head. Gold labels are not touched."""
    model = densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 12)
    incoming = keras_feature_state(path)
    current = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    shape_errors: list[str] = []
    for key, tensor in incoming.items():
        if key not in current:
            continue
        if tuple(current[key].shape) != tuple(tensor.shape):
            shape_errors.append(f"{key} keras {tuple(tensor.shape)} torch {tuple(current[key].shape)}")
            continue
        matched[key] = tensor
    if shape_errors:
        raise ValueError("shape mismatch: " + "; ".join(shape_errors[:8]))
    missing, unexpected = model.load_state_dict(matched, strict=False)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    report = {
        "n_loaded": len(matched),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "classifier_out": int(model.classifier.out_features),
        "features_frozen": all(not p.requires_grad for p in model.features.parameters()),
        "head_trainable": any(p.requires_grad for p in model.classifier.parameters()),
    }
    return model, report

"""8-study timing math for a frozen RadImageNet DenseNet121 encode.

The full 4349-study train is not started here. Gold expert rows are excluded
from the probe sample so they cannot enter a loss.
"""


import numpy as np
import pandas as pd


PROBE_STUDIES = 8
IMAGES_PER_STUDY = 18
N_ENCODE_STUDIES = 4407
ENCODE_ABORT_S = 2.8 * 3600.0


def probe_study_ids(
    train: pd.DataFrame,
    *,
    n: int = PROBE_STUDIES,
    seed: int = 2026,
) -> tuple[str, ...]:
    """Non-gold study IDs. The returned set is disjoint from the expert 58."""
    gold = set(expert_gold_ids(train))
    frame = train.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    pool = [uid for uid in frame[ID_COLUMN].tolist() if uid not in gold]
    if len(pool) < int(n):
        raise ValueError(f"need {n} non-gold studies, have {len(pool)}")
    rng = np.random.default_rng(int(seed))
    chosen_at = np.sort(rng.choice(len(pool), size=int(n), replace=False))
    chosen = tuple(pool[int(i)] for i in chosen_at.tolist())
    if set(chosen) & gold:
        raise ValueError("gold UID leaked into the DenseNet probe sample")
    return chosen


def projected_encode_seconds(
    elapsed_s: float,
    n_images: int,
    *,
    n_studies: int = N_ENCODE_STUDIES,
    images_per_study: int = IMAGES_PER_STUDY,
) -> float:
    """Scale a measured image batch onto the full encode set."""
    elapsed = float(elapsed_s)
    images = int(n_images)
    if elapsed <= 0.0 or images <= 0:
        raise ValueError("elapsed time and image count must be positive")
    return elapsed * (int(n_studies) * int(images_per_study)) / images


def encode_probe_aborts(
    elapsed_s: float,
    n_images: int,
    *,
    limit_s: float = ENCODE_ABORT_S,
) -> bool:
    """True when the projected full encode is strictly above 2.8 h."""
    return projected_encode_seconds(elapsed_s, n_images) > float(limit_s)

"""Frozen DenseNet121 head fit. Gold 58 and the mill-870 holdout stay out of the loss."""


import numpy as np
import torch
from torch import nn


HEAD_EPOCHS = 25
HEAD_LR = 1e-2
HEAD_WEIGHT_DECAY = 0.05
HEAD_SEED = 2026


def densenet_id_split(train, v4) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return fit, mill-870 holdout, and gold ids. The three sets are disjoint."""
    split = make_steven_v4_split(train, v4)
    fit_idx, hold_idx = llm_soft_holdout_indices(split.n_train, frac=0.2, seed=2026)
    fit_ids = tuple(split.train_ids[int(i)] for i in fit_idx.tolist())
    hold_ids = tuple(split.train_ids[int(i)] for i in hold_idx.tolist())
    gold_ids = tuple(split.val_ids)
    if set(fit_ids) & set(hold_ids):
        raise ValueError("holdout leaked into the DenseNet fit")
    if set(fit_ids) & set(gold_ids) or set(hold_ids) & set(gold_ids):
        raise ValueError("gold leaked into the DenseNet fit or holdout")
    return fit_ids, hold_ids, gold_ids


def fit_linear_head(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    w_fit: np.ndarray,
    x_eval: np.ndarray,
    *,
    epochs: int = HEAD_EPOCHS,
    lr: float = HEAD_LR,
    weight_decay: float = HEAD_WEIGHT_DECAY,
    seed: int = HEAD_SEED,
) -> np.ndarray:
    """BCE-with-logits on soft targets. Returns probabilities for x_eval."""
    if x_fit.shape[0] != y_fit.shape[0] or y_fit.shape != w_fit.shape:
        raise ValueError("fit rows must match targets and weights")
    if x_fit.shape[1] != x_eval.shape[1]:
        raise ValueError("eval width must match fit width")
    generator = torch.Generator().manual_seed(int(seed))
    layer = nn.Linear(int(x_fit.shape[1]), int(y_fit.shape[1]))
    nn.init.xavier_uniform_(layer.weight, generator=generator)
    nn.init.zeros_(layer.bias)
    opt = torch.optim.AdamW(layer.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    x = torch.as_tensor(x_fit, dtype=torch.float32)
    y = torch.as_tensor(np.clip(y_fit, 0.0, 1.0), dtype=torch.float32)
    w = torch.as_tensor(w_fit, dtype=torch.float32)
    n = int(x.shape[0])
    layer.train()
    for _ in range(int(epochs)):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, 256):
            idx = perm[start : start + 256]
            opt.zero_grad()
            loss = (loss_fn(layer(x[idx]), y[idx]) * w[idx]).mean()
            loss.backward()
            opt.step()
    layer.eval()
    with torch.no_grad():
        logits = layer(torch.as_tensor(x_eval, dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy()
    return np.asarray(probs, dtype=np.float64)


def gold_gate(y_gold: np.ndarray, scores: np.ndarray) -> tuple[float, bool]:
    """Macro AUC on expert gold. Abort when it is strictly above 0.94."""
    macro = macro_roc_auc(y_gold, scores)
    return macro, should_abort_gold_overfit(macro)


import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
print("nvidia-smi:", subprocess.getoutput("nvidia-smi -L"), flush=True)
print(
    "[dn121-head] PRIVATE frozen RadImageNet-DenseNet121 + new 12-way head. "
    "Gold-58 out of the loss. Mill-870 out of the loss. DO NOT SUBMIT. keep 55779066.",
    flush=True,
)

HARD_LIMIT_S = int(ENCODE_ABORT_S)


def _watchdog() -> None:
    time.sleep(HARD_LIMIT_S)
    print(f"[dn121-head] WATCHDOG {HARD_LIMIT_S}s", flush=True)
    os._exit(99)


threading.Thread(target=_watchdog, daemon=True, name="dn121-head-watchdog").start()


def _comp_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    for path in candidates:
        if (path / "train.csv").is_file() and (path / "train_series").is_dir():
            return path
    base = Path("/kaggle/input")
    for path in base.rglob("train.csv"):
        if (path.parent / "train_series").is_dir():
            return path.parent
    raise FileNotFoundError("competition train_series not mounted")


def _find_named(name: str) -> Path | None:
    base = Path("/kaggle/input")
    skip = {"train_series", "test_series", "train_images", "test_images"}
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [item for item in dirnames if item not in skip and not item.startswith("1.2.")]
        if name in filenames:
            return Path(dirpath) / name
    return None


def _study_batch(root: Path, uid: str, k: int = IMAGES_PER_STUDY) -> torch.Tensor | None:
    folder = root / "train_series" / uid
    files = sorted(path for path in folder.rglob("*") if path.suffix.lower() == ".dcm")
    if not files:
        return None
    if len(files) > k:
        pick = np.linspace(0, len(files) - 1, k).astype(int)
        files = [files[int(i)] for i in pick]
    images = []
    for path in files:
        array = np.asarray(pydicom.dcmread(str(path), force=True).pixel_array, dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        low, high = np.percentile(array, [1, 99])
        scale = max(float(high - low), 1e-6)
        array = np.clip((array - float(low)) / scale, 0.0, 1.0).astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        images.append(tensor.repeat(1, 3, 1, 1))
    if not images:
        return None
    return torch.cat(images, dim=0)


def _encode(model, device, comp: Path, ids: list[str]) -> tuple[np.ndarray, int]:
    rows = []
    missing = 0
    started = time.perf_counter()
    n_images = 0
    for n_done, uid in enumerate(ids, start=1):
        batch = _study_batch(comp, uid)
        if batch is None:
            missing += 1
            rows.append(np.full(1024, np.nan, dtype=np.float32))
        else:
            with torch.no_grad():
                feats = model.features(batch.to(device).float())
                pooled = F.adaptive_avg_pool2d(F.relu(feats), (1, 1)).flatten(1).mean(dim=0)
            rows.append(pooled.detach().cpu().numpy().astype(np.float32))
            n_images += int(batch.shape[0])
            if device.type == "cuda":
                torch.cuda.synchronize()
        if n_done == PROBE_STUDIES and n_images > 0:
            elapsed = time.perf_counter() - started
            if encode_probe_aborts(elapsed, n_images):
                print(
                    f"[dn121-head] ABORT early projected encode > 2.8h "
                    f"elapsed_s={elapsed:.1f} images={n_images}",
                    flush=True,
                )
                raise SystemExit(2)
        if n_done % 200 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[dn121-head] encoded {n_done}/{len(ids)} images={n_images} "
                f"missing={missing} elapsed_s={elapsed:.1f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    print(
        f"[dn121-head] encode done n={len(ids)} images={n_images} missing={missing} "
        f"elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return np.stack(rows), missing


def _table(ids, scores) -> pd.DataFrame:
    frame = pd.DataFrame(scores, columns=list(FINDINGS))
    frame.insert(0, "StudyInstanceUID", list(ids))
    return frame


def main() -> None:
    comp = _comp_root()
    v4_path = _find_named("llm_labels_v4_blend.csv")
    if v4_path is None:
        raise SystemExit("[dn121-head] steven v4 labels missing")
    h5_path = _find_named("RadImageNet-DenseNet121_notop.h5")
    if h5_path is None:
        raise SystemExit("[dn121-head] densenet h5 missing")
    train = pd.read_csv(comp / "train.csv", usecols=["StudyInstanceUID", *FINDINGS])
    v4 = pd.read_csv(v4_path, dtype={"StudyInstanceUID": str})
    fit_ids, hold_ids, gold_ids = densenet_id_split(train, v4)
    print(
        f"[dn121-head] fit_n={len(fit_ids)} holdout_n={len(hold_ids)} gold_n={len(gold_ids)} "
        f"gold_in_fit={len(set(fit_ids) & set(gold_ids))} "
        f"holdout_in_fit={len(set(fit_ids) & set(hold_ids))}",
        flush=True,
    )
    model, report = load_radimagenet_densenet121(h5_path)
    print(
        f"[dn121-head] loaded n={report['n_loaded']} unexpected={len(report['unexpected'])} "
        f"frozen={report['features_frozen']} head={report['classifier_out']}",
        flush=True,
    )
    if report["unexpected"] or not report["features_frozen"]:
        raise SystemExit("[dn121-head] weight load failed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    order = list(fit_ids) + list(hold_ids) + list(gold_ids)
    encoded, missing = _encode(model, device, comp, order)
    by_id = {uid: encoded[i] for i, uid in enumerate(order)}

    def stack(ids):
        return np.stack([by_id[uid] for uid in ids])

    v4_idx = v4.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    y_fit = v4_idx.loc[list(fit_ids), list(FINDINGS)].to_numpy(dtype=np.float64)
    pw_path = _find_named("report_labels_v2.csv")
    if pw_path is None:
        print("[dn121-head] pilkwang conf missing; weight=1", flush=True)
        w_fit = np.ones_like(y_fit)
    else:
        pw = pd.read_csv(pw_path, dtype={"StudyInstanceUID": str})
        _targets, weights = grade_soft_targets(v4, pw)
        w_idx = weights.set_index("StudyInstanceUID")
        w_fit = w_idx.reindex(list(fit_ids))[list(FINDINGS)].to_numpy(dtype=np.float64)
        w_fit = np.where(np.isfinite(w_fit), w_fit, 1.0)
    x_fit = stack(fit_ids)
    if not np.isfinite(x_fit).all() or not np.isfinite(y_fit).all():
        raise SystemExit("[dn121-head] non-finite fit rows")
    x_hold = stack(hold_ids)
    x_gold = stack(gold_ids)
    y_gold = (
        train.drop_duplicates("StudyInstanceUID")
        .assign(StudyInstanceUID=lambda frame: frame["StudyInstanceUID"].astype(str))
        .set_index("StudyInstanceUID")
        .loc[list(gold_ids), list(FINDINGS)]
        .to_numpy(dtype=np.float64)
    )
    eval_scores = fit_linear_head(x_fit, y_fit, w_fit, np.concatenate([x_gold, x_hold]))
    gold_scores = eval_scores[: len(gold_ids)]
    hold_scores = eval_scores[len(gold_ids) :]
    macro, abort = gold_gate(y_gold, gold_scores)
    print(f"[dn121-head] gold_macro={macro:.4f} abort={abort} threshold=0.94", flush=True)
    work = Path("/kaggle/working")
    work.mkdir(parents=True, exist_ok=True)
    sub = work / "submission.csv"
    if sub.is_file():
        sub.unlink()
    log = (
        f"fit_n={len(fit_ids)}\nholdout_n={len(hold_ids)}\ngold_n={len(gold_ids)}\n"
        f"gold_in_fit=0\nholdout_in_fit=0\nmissing_images={missing}\n"
        f"gold_macro={macro:.6f}\nabort={abort}\nDO NOT SUBMIT\n"
    )
    (work / "protocol_log.txt").write_text(log, encoding="utf-8")
    _table(gold_ids, gold_scores).to_csv(work / "densenet_gold58.csv", index=False)
    if abort:
        print("[dn121-head] ABORT gold macro > 0.94. No holdout license. DO NOT SUBMIT.", flush=True)
        raise SystemExit(0)
    _table(hold_ids, hold_scores).to_csv(work / "densenet_holdout870.csv", index=False)
    (work / "DO_NOT_SUBMIT_HOLDOUT.txt").write_text(
        "densenet_holdout870.csv is the seed-2026 train holdout, not the hidden test.\n"
        "Keep 55779066 until a local mill-870 join licenses a TEST overlay.\n",
        encoding="utf-8",
    )
    print(
        f"[dn121-head] wrote densenet_holdout870.csv rows={len(hold_ids)} missing_images={missing}. "
        "DO NOT SUBMIT. keep 55779066.",
        flush=True,
    )


if __name__ == "__main__":
    main()
