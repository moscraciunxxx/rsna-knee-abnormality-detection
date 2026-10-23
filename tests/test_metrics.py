"""Drive the shipped macro-averaged ROC AUC (12 findings)."""

import numpy as np

from rsna_knee.metrics import macro_roc_auc, per_finding_roc_auc
from rsna_knee.schema import FINDINGS


def _mixed_ranking_labels(n=40, n_findings=12, seed=7):
    """Synthetic multilabel with mixed ranking: better than chance, not perfect."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(n, n_findings)).astype(np.float64)
    # Guarantee both classes per finding so AUC is defined.
    y_true[0, :] = 0.0
    y_true[1, :] = 1.0
    noise = rng.normal(0.0, 0.45, size=y_true.shape)
    y_score = np.clip(0.25 + 0.5 * y_true + noise, 0.0, 1.0)
    return y_true, y_score


def test_macro_roc_auc_is_mean_of_twelve_per_finding_aucs():
    y_true, y_score = _mixed_ranking_labels()
    per = per_finding_roc_auc(y_true, y_score)
    macro = macro_roc_auc(y_true, y_score)
    assert per.shape == (len(FINDINGS),)
    assert np.isfinite(per).all()
    np.testing.assert_allclose(macro, float(np.mean(per)))


def test_macro_roc_auc_mixed_ranking_beats_chance_and_is_not_perfect():
    y_true, y_score = _mixed_ranking_labels()
    score = macro_roc_auc(y_true, y_score)
    chance = 0.5
    perfect = 1.0
    assert score > chance
    assert score < perfect


def test_perfect_ranking_is_one_and_reversed_is_below_chance():
    y_true = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], dtype=float)
    y_true = np.tile(y_true, (1, 6))  # 12 columns
    y_good = y_true.copy()
    y_bad = 1.0 - y_true
    good = macro_roc_auc(y_true, y_good)
    bad = macro_roc_auc(y_true, y_bad)
    assert good == 1.0
    assert bad < 0.5


def test_rare_soft_labels_keep_unk_off_zero_and_downweight_silence():
    """Real pilkwang + steven v4 tables. UNK stays a soft score, not a zero."""
    from pathlib import Path

    import pandas as pd

    from rsna_knee.schema import ID_COLUMN
    from rsna_knee.soft_labels import (
        RARE_FINDINGS,
        audit_rare_labels,
        grade_soft_targets,
        harden_unk_to_zero,
        non_gold_train_ids,
    )
    from rsna_knee.v4_split import expert_gold_ids

    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(root / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv", dtype={ID_COLUMN: str})
    pw = pd.read_csv(root / ".cache" / "llm_labels" / "report_labels_v2.csv", dtype={ID_COLUMN: str})
    assert "Report" not in v4.columns
    assert "Report" not in pw.columns

    targets, weights = grade_soft_targets(v4, pw)
    assert list(targets.columns) == [ID_COLUMN, *FINDINGS]
    assert "Baker's" in targets.columns
    v4_ids = v4.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)
    aligned = targets.set_index(ID_COLUMN).loc[v4_ids.index]
    for name in FINDINGS:
        np.testing.assert_allclose(
            aligned[name].to_numpy(dtype=float),
            pd.to_numeric(v4_ids[name], errors="coerce").to_numpy(dtype=float),
        )
        assert aligned[name].min() > 0.0 or aligned[name].max() < 1.0

    hardened = harden_unk_to_zero(pw).set_index(ID_COLUMN)
    pw_ids = pw.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)
    weight_ids = weights.set_index(ID_COLUMN)
    for name in RARE_FINDINGS:
        unk = pw_ids[f"{name}__verdict"].astype(str).eq("UNK")
        assert int(unk.sum()) > 0
        soft = aligned.loc[pw_ids.index[unk], name].to_numpy(dtype=float)
        killed = hardened.loc[pw_ids.index[unk], name].to_numpy(dtype=float)
        assert np.any(soft != killed)
        conf = pd.to_numeric(pw_ids.loc[unk, f"{name}__conf"], errors="coerce").clip(0.0, 1.0)
        np.testing.assert_allclose(weight_ids.loc[pw_ids.index[unk], name].to_numpy(dtype=float), conf.to_numpy(dtype=float))

    gold = expert_gold_ids(train)
    train_ids = non_gold_train_ids(targets[ID_COLUMN], gold)
    assert set(train_ids).isdisjoint(gold)
    assert len(train_ids) == len(targets) - len(set(targets[ID_COLUMN]).intersection(gold))

    audit = audit_rare_labels(train, v4, pw)
    rates = audit["unk_rate"]
    shares = audit["unk_loss_share"]
    assert rates["Synovitis"] > rates["Fracture"] > rates["Baker's"] > rates["MCL"]
    assert shares["Synovitis"] < rates["Synovitis"]
    assert shares["Fracture"] < rates["Fracture"]
    assert audit["gold_auc_hardened"]["Fracture"] < audit["gold_auc_pilkwang"]["Fracture"]
    assert audit["gold_auc_hardened"]["Baker's"] < audit["gold_auc_pilkwang"]["Baker's"]
    assert audit["n_gold"] == 58
    assert audit["n_gold_in_v4"] == 58
    assert audit["n_gold_in_pilkwang"] == 57
    assert audit["n_train"] == audit["n_v4"] - 58

    from rsna_knee.soft_labels import confidence_graded_targets, rare_gold_aucs

    toy_v4 = pd.DataFrame({
        ID_COLUMN: ["s1", "s2"],
        **{name: [0.2, 0.8] for name in FINDINGS},
    })
    toy_pw = toy_v4.copy()
    for name in FINDINGS:
        toy_pw[f"{name}__verdict"] = ["UNK", "POS"]
        toy_pw[f"{name}__conf"] = [0.1, 1.0]
        toy_pw[name] = [0.0, 0.6]
    toy = confidence_graded_targets(toy_v4, toy_pw).set_index(ID_COLUMN)
    assert toy.loc["s1", "Fracture"] == 0.2
    assert toy.loc["s2", "Baker's"] == 0.6

    graded = confidence_graded_targets(v4, pw).set_index(ID_COLUMN)
    for name in RARE_FINDINGS:
        unk = pw_ids[f"{name}__verdict"].astype(str).eq("UNK")
        unk_ids = pw_ids.index[unk]
        common = graded.index.intersection(unk_ids).intersection(aligned.index)
        np.testing.assert_allclose(
            graded.loc[common, name].to_numpy(dtype=float),
            aligned.loc[common, name].to_numpy(dtype=float),
        )
    auc_v4 = rare_gold_aucs(train, targets)
    auc_graded = rare_gold_aucs(train, graded.reset_index())
    assert set(auc_graded) == set(RARE_FINDINGS)
    # Known pilkwang mentions blended into v4 do not beat v4 on Fracture gold.
    # Keep v4 as the training target. UNK rows above stayed equal to v4.
    assert auc_graded["Fracture"] < auc_v4["Fracture"]
    assert auc_v4["MCL"] > 0.5

    from rsna_knee.soft_labels import gold_finding_aucs, rank_average_score_tables

    full_v4 = gold_finding_aucs(train, targets)
    for name in RARE_FINDINGS:
        assert full_v4[name] == auc_v4[name]
    assert full_v4["n_gold"] == 58.0
    assert abs(full_v4["macro"] - sum(full_v4[name] for name in FINDINGS) / 12.0) < 1e-12

    left = pd.DataFrame({ID_COLUMN: ["s1", "s2"], **{name: [0.2, 0.8] for name in FINDINGS}})
    right = left.copy()
    right["Fracture"] = [0.9, 0.1]
    averaged = rank_average_score_tables([left, right]).set_index(ID_COLUMN)
    assert averaged.loc["s1", "ACL"] == 0.5
    assert averaged.loc["s2", "ACL"] == 1.0
    assert averaged.loc["s1", "Fracture"] == 0.75
    assert averaged.loc["s2", "Fracture"] == 0.75
    real_avg = rank_average_score_tables([v4, pw])
    assert list(real_avg.columns) == [ID_COLUMN, *FINDINGS]
    assert "Baker's" in real_avg.columns
    assert len(real_avg) > 4000
    assert float(real_avg[list(FINDINGS)].min().min()) > 0.0
    assert float(real_avg[list(FINDINGS)].max().max()) <= 1.0

    from rsna_knee.soft_labels import FRACTURE_GRAFT_FINDINGS, graft_finding_scores

    assert FRACTURE_GRAFT_FINDINGS == ("Fracture",)
    shared = set(expert_gold_ids(train)) & set(v4[ID_COLUMN].astype(str)) & set(pw[ID_COLUMN].astype(str))
    assert len(shared) == 57
    v4_shared = v4[v4[ID_COLUMN].isin(shared)]
    grafted = graft_finding_scores(v4, pw, FRACTURE_GRAFT_FINDINGS)
    graft_shared = grafted[grafted[ID_COLUMN].isin(shared)]
    auc_base = gold_finding_aucs(train, v4_shared)
    auc_graft = gold_finding_aucs(train, graft_shared)
    assert auc_graft["n_gold"] == 57.0
    assert auc_graft["Fracture"] > auc_base["Fracture"] + 0.05
    assert auc_graft["macro"] > auc_base["macro"] + 0.005
    assert auc_graft["Synovitis"] == auc_base["Synovitis"]
    assert auc_graft["MCL"] == auc_base["MCL"]
    replaced = graft_finding_scores(v4, pw, FINDINGS)
    auc_replaced = gold_finding_aucs(train, replaced[replaced[ID_COLUMN].isin(shared)])
    assert auc_replaced["macro"] < auc_base["macro"]
    auc_avg = gold_finding_aucs(train, real_avg)
    assert auc_avg["macro"] - auc_base["macro"] < 0.005
    graded_macro = gold_finding_aucs(train, graded.reset_index())
    assert graded_macro["macro"] < full_v4["macro"]
    unk_fracture = pw_ids[pw_ids["Fracture__verdict"].astype(str).eq("UNK")]
    unk_id = str(unk_fracture.index[0])
    copied = grafted.set_index(ID_COLUMN).loc[unk_id, "Fracture"]
    assert copied == float(pw_ids.loc[unk_id, "Fracture"])
    assert copied != 0.0

    from rsna_knee.soft_labels import fracture_graft_train_frame

    train_frame = fracture_graft_train_frame(v4, pw, train)
    assert "Report" not in train_frame.columns
    assert list(train_frame.columns) == [ID_COLUMN, *FINDINGS]
    assert set(train_frame[ID_COLUMN].astype(str)).isdisjoint(gold)
    assert len(train_frame) == len(v4.drop_duplicates(ID_COLUMN)) - 58
    train_idx = train_frame.set_index(ID_COLUMN)
    v4_idx = v4.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)
    pw_score = pw.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)
    sample_id = train_idx.index[0]
    assert float(train_idx.loc[sample_id, "ACL"]) == float(v4_idx.loc[sample_id, "ACL"])
    assert float(train_idx.loc[sample_id, "Fracture"]) == float(pw_score.loc[sample_id, "Fracture"])

    from rsna_knee.soft_labels import fracture_graft_train_bundle

    bundle_y, bundle_w = fracture_graft_train_bundle(v4, pw, train)
    assert list(bundle_y[ID_COLUMN]) == list(bundle_w[ID_COLUMN])
    assert set(bundle_w[ID_COLUMN].astype(str)).isdisjoint(gold)
    assert len(bundle_y) == len(train_frame)
    bundle_idx = bundle_y.set_index(ID_COLUMN)
    weight_idx = bundle_w.set_index(ID_COLUMN)
    assert float(bundle_idx.loc[sample_id, "Fracture"]) == float(pw_score.loc[sample_id, "Fracture"])
    assert float(bundle_idx.loc[sample_id, "ACL"]) == float(v4_idx.loc[sample_id, "ACL"])
    unk_in_train = [uid for uid in unk_fracture.index.astype(str) if uid in bundle_idx.index]
    assert unk_in_train
    unk_train_id = unk_in_train[0]
    assert float(bundle_idx.loc[unk_train_id, "Fracture"]) == float(pw_score.loc[unk_train_id, "Fracture"])
    assert float(bundle_idx.loc[unk_train_id, "Fracture"]) != 0.0
    assert float(weight_idx.loc[unk_train_id, "Fracture"]) == float(
        pd.to_numeric(pw_score.loc[unk_train_id, "Fracture__conf"], errors="coerce")
    )

    from rsna_knee.v4_split import HOLDOUT_MACRO_STOP, frozen_head_gate, should_abort_gold_overfit

    assert HOLDOUT_MACRO_STOP == 0.745
    assert should_abort_gold_overfit(0.95) is True
    assert frozen_head_gate(0.95, 0.99) == "abort"
    assert should_abort_gold_overfit(0.94) is False
    assert frozen_head_gate(0.94, 0.80) == "continue"
    assert frozen_head_gate(0.80, HOLDOUT_MACRO_STOP) == "stop"
    assert frozen_head_gate(0.80, HOLDOUT_MACRO_STOP + 1e-4) == "continue"
    assert frozen_head_gate(float("nan"), 0.99) == "stop"

    from rsna_knee.soft_labels import graft_label_clears, graft_versus_v4_gold_delta

    delta = graft_versus_v4_gold_delta(train, v4, pw, n_draws=400, seed=2026)
    assert delta["n"] == 58.0
    assert delta["n_draws"] == 400.0
    assert delta["lo"] <= delta["point"] <= delta["hi"]
    assert delta["point"] > 0.005
    assert graft_label_clears(delta) is True


def test_bootstrap_macro_delta_is_zero_when_the_scores_match():
    from rsna_knee.soft_labels import bootstrap_macro_delta, graft_label_clears

    y = np.zeros((4, 12), dtype=np.float64)
    y[1:, :] = 1.0
    y[0, :] = 0.0
    scores = np.tile(np.array([[0.1], [0.4], [0.6], [0.9]], dtype=np.float64), (1, 12))
    delta = bootstrap_macro_delta(y, scores, scores, n_draws=30, seed=2026)
    assert delta["point"] == 0.0
    assert delta["lo"] == 0.0
    assert delta["hi"] == 0.0
    assert graft_label_clears(delta) is False


def test_graft_fit_matrices_copy_fracture_and_reject_gold(tmp_path):
    import pytest
    from pathlib import Path

    import pandas as pd

    from rsna_knee.schema import FINDINGS, ID_COLUMN
    from rsna_knee.soft_labels import fracture_graft_train_bundle, graft_fit_matrices
    from rsna_knee.v4_split import expert_gold_ids

    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(root / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv", dtype={ID_COLUMN: str})
    pw = pd.read_csv(root / ".cache" / "llm_labels" / "report_labels_v2.csv", dtype={ID_COLUMN: str})
    if "Report" in train.columns:
        train = train.drop(columns=["Report"])
    targets, weights = fracture_graft_train_bundle(v4, pw, train)
    fit_ids = targets[ID_COLUMN].astype(str).head(40).tolist()
    y, w = graft_fit_matrices(v4, pw, train, fit_ids)
    assert y.shape == (40, len(FINDINGS))
    assert w.shape == y.shape
    assert np.isfinite(y).all()
    assert np.isfinite(w).all()
    fracture = FINDINGS.index("Fracture")
    acl = FINDINGS.index("ACL")
    y_idx = targets.set_index(ID_COLUMN)
    w_idx = weights.set_index(ID_COLUMN)
    pw_idx = pw.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)
    v4_idx = v4.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)
    np.testing.assert_allclose(y[:, fracture], y_idx.loc[fit_ids, "Fracture"].to_numpy(dtype=float))
    np.testing.assert_allclose(y[:, acl], v4_idx.loc[fit_ids, "ACL"].to_numpy(dtype=float))
    np.testing.assert_allclose(y[:, fracture], pw_idx.loc[fit_ids, "Fracture"].to_numpy(dtype=float))
    np.testing.assert_allclose(w[:, fracture], w_idx.loc[fit_ids, "Fracture"].to_numpy(dtype=float))
    gold = expert_gold_ids(train)
    with pytest.raises(ValueError, match="gold UID"):
        graft_fit_matrices(v4, pw, train, [gold[0], fit_ids[0]])

    from rsna_knee.graft_head_export import graft_head_frames, graft_head_splits
    from rsna_knee.graft_head_kernel import export_directory

    fit_ids, hold_ids, gold_ids = graft_head_splits(train, v4)
    assert set(fit_ids).isdisjoint(gold_ids)
    assert set(hold_ids).isdisjoint(gold_ids)
    assert set(fit_ids).isdisjoint(hold_ids)
    assert len(fit_ids) + len(hold_ids) == len(set(fit_ids) | set(hold_ids))
    frames, frame_w, returned_hold = graft_head_frames(v4, pw, train)
    assert list(frames.columns) == [ID_COLUMN, *FINDINGS]
    assert "Baker's" in frames.columns
    assert "Report" not in frames.columns
    assert set(frames[ID_COLUMN].astype(str)).isdisjoint(gold_ids)
    assert list(returned_hold) == list(hold_ids)
    assert len(frames) == len(fit_ids)
    data = tmp_path / "data"
    data.mkdir()
    train.to_csv(data / "train.csv", index=False)
    v4.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    pw.to_csv(data / "report_labels_v2.csv", index=False)
    work = tmp_path / "work"
    work.mkdir()
    (work / "submission.csv").write_text("junk", encoding="utf-8")
    fit_n, hold_n = export_directory(data, work)
    assert fit_n == len(fit_ids)
    assert hold_n == len(hold_ids)
    assert not (work / "submission.csv").exists()
    written = pd.read_csv(work / "graft_fit_targets.csv", dtype={ID_COLUMN: str})
    assert list(written.columns) == [ID_COLUMN, *FINDINGS]
    assert "Report" not in written.columns
