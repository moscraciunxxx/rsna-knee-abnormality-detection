"""Drive shipped per-finding fusion. The 0.917 uniform 0.50 all-12 blend must fail."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.fusion import (
    DINO3S_ENCODER,
    DINO3S_REPLACE_FINDINGS,
    DINO3S_REPLACE_WEIGHT,
    GOLDFOLD_ENCODER,
    GOLDFOLD_N_GOLD,
    GOLDFOLD_OVERLAY_WEIGHT,
    R336OV_ENCODER,
    R336OV_OVERLAY_FINDINGS,
    R336OV_OVERLAY_WEIGHT,
    is_ruled_out_r336ov_family,
    JOINTLINE_CROP_MM,
    JOINTLINE_OVERLAY_FINDINGS,
    JOINTLINE_OVERLAY_WEIGHT,
    LOGFIX_OVERLAY_FINDINGS,
    LOGFIX_OVERLAY_WEIGHT,
    TTAMAX_DINO_POOL,
    TTAMAX_OVERLAY_FINDINGS,
    TTAMAX_OVERLAY_WEIGHT,
    CONVNEXT_OOF_K7_FINDINGS,
    CONVNEXT_OOF_K7_HI_WEIGHT,
    CONVNEXT_OOF_K7_WEIGHT,
    convnext_oof_k7_hi_weights,
    convnext_oof_k7_weights,
    convnext_oof_unequal_weights,
    V41_SATURATED_FINDINGS,
    FusionError,
    is_ruled_out_dino3s_replace_family,
    is_ruled_out_jointline_mill_radimagenet_family,
    is_ruled_out_public_native384_family,
    is_uniform_all_finding_overlay,
    overlay_finding_set,
    fail_closed_rank_blend,
    overlay_weights_from_findings,
    per_finding_rank_blend,
)
from rsna_knee.guards import kernel_source_text
from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS


def _table(seed: int, n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scores = rng.random((n, len(FINDINGS)))
    frame = pd.DataFrame(scores, columns=list(FINDINGS))
    frame.insert(0, ID_COLUMN, [f"s{i}" for i in range(n)])
    return frame[list(SUBMISSION_COLUMNS)]


def test_uniform_050_all_twelve_is_the_cnn_tri_pattern():
    weights = overlay_weights_from_findings(FINDINGS, 0.50)
    assert is_uniform_all_finding_overlay(weights, min_weight=0.5)
    with pytest.raises(FusionError, match="CNN-tri 0.917"):
        per_finding_rank_blend(_table(1), _table(2), weights)


def test_v41_three_target_050_is_allowed_and_leaves_other_columns():
    anchor = _table(3)
    overlay = _table(4)
    weights = overlay_weights_from_findings(V41_SATURATED_FINDINGS, 0.50)
    assert not is_uniform_all_finding_overlay(weights, min_weight=0.5)
    out = per_finding_rank_blend(anchor, overlay, weights)
    assert list(out.columns) == list(SUBMISSION_COLUMNS)
    assert "Baker's" in out.columns
    for name in FINDINGS:
        if name in V41_SATURATED_FINDINGS:
            assert not np.allclose(out[name].to_numpy(), anchor[name].to_numpy())
        else:
            np.testing.assert_array_equal(out[name].to_numpy(), anchor[name].to_numpy())


def test_logfix_overlay_skips_v41_saturated_targets_and_is_not_all_twelve():
    weights = overlay_weights_from_findings(LOGFIX_OVERLAY_FINDINGS, LOGFIX_OVERLAY_WEIGHT)
    assert set(LOGFIX_OVERLAY_FINDINGS).isdisjoint(V41_SATURATED_FINDINGS)
    assert set(LOGFIX_OVERLAY_FINDINGS) != set(FINDINGS)
    assert "Baker's" in LOGFIX_OVERLAY_FINDINGS
    assert not is_uniform_all_finding_overlay(weights, min_weight=0.5)
    anchor = _table(5)
    overlay = _table(6)
    out = per_finding_rank_blend(anchor, overlay, weights)
    for name in V41_SATURATED_FINDINGS:
        np.testing.assert_array_equal(out[name].to_numpy(), anchor[name].to_numpy())
    for name in LOGFIX_OVERLAY_FINDINGS:
        assert not np.allclose(out[name].to_numpy(), anchor[name].to_numpy())


def test_v41_and_logfix_native384_sets_are_ruled_out_as_new_family():
    assert is_ruled_out_public_native384_family(V41_SATURATED_FINDINGS)
    assert is_ruled_out_public_native384_family(LOGFIX_OVERLAY_FINDINGS)
    assert not is_ruled_out_public_native384_family(JOINTLINE_OVERLAY_FINDINGS)
    assert "Synovitis" in JOINTLINE_OVERLAY_FINDINGS
    assert JOINTLINE_CROP_MM < 130.0
    weights = overlay_weights_from_findings(
        JOINTLINE_OVERLAY_FINDINGS, JOINTLINE_OVERLAY_WEIGHT
    )
    assert overlay_finding_set(weights) == frozenset(JOINTLINE_OVERLAY_FINDINGS)
    assert not is_uniform_all_finding_overlay(weights, min_weight=0.5)
    anchor = _table(7)
    overlay = _table(8)
    out = per_finding_rank_blend(anchor, overlay, weights)
    for name in FINDINGS:
        if name in JOINTLINE_OVERLAY_FINDINGS:
            assert not np.allclose(out[name].to_numpy(), anchor[name].to_numpy())
        else:
            np.testing.assert_array_equal(out[name].to_numpy(), anchor[name].to_numpy())


def test_jointline_90mm_mill_radimagenet_is_ruled_out_as_new_family():
    assert is_ruled_out_jointline_mill_radimagenet_family(
        JOINTLINE_OVERLAY_FINDINGS, crop_mm=JOINTLINE_CROP_MM
    )
    assert is_ruled_out_jointline_mill_radimagenet_family(JOINTLINE_OVERLAY_FINDINGS)
    assert not is_ruled_out_jointline_mill_radimagenet_family(DINO3S_REPLACE_FINDINGS)
    assert not is_ruled_out_public_native384_family(DINO3S_REPLACE_FINDINGS)
    assert DINO3S_REPLACE_WEIGHT == 1.0
    assert DINO3S_ENCODER == "dinov3-vits16plus"
    assert "Synovitis" not in DINO3S_REPLACE_FINDINGS
    assert "Medial Meniscus" not in DINO3S_REPLACE_FINDINGS
    weights = overlay_weights_from_findings(DINO3S_REPLACE_FINDINGS, DINO3S_REPLACE_WEIGHT)
    assert not is_uniform_all_finding_overlay(weights, min_weight=0.5)
    assert overlay_finding_set(weights) == frozenset(DINO3S_REPLACE_FINDINGS)
    anchor = _table(9)
    overlay = _table(10)
    out = per_finding_rank_blend(anchor, overlay, weights)
    for name in FINDINGS:
        if name in DINO3S_REPLACE_FINDINGS:
            assert not np.allclose(out[name].to_numpy(), anchor[name].to_numpy())
        else:
            np.testing.assert_array_equal(out[name].to_numpy(), anchor[name].to_numpy())


def test_dino3s_weight_one_mcl_bakers_is_ruled_out_as_new_family():
    assert is_ruled_out_dino3s_replace_family(DINO3S_REPLACE_FINDINGS)
    assert is_ruled_out_dino3s_replace_family(
        DINO3S_REPLACE_FINDINGS, weight=DINO3S_REPLACE_WEIGHT
    )
    assert not is_ruled_out_dino3s_replace_family(TTAMAX_OVERLAY_FINDINGS)
    assert not is_ruled_out_dino3s_replace_family(
        DINO3S_REPLACE_FINDINGS, weight=TTAMAX_OVERLAY_WEIGHT
    )
    assert TTAMAX_OVERLAY_WEIGHT < 0.50
    assert TTAMAX_DINO_POOL == "max"
    assert "PF OA" in TTAMAX_OVERLAY_FINDINGS
    assert "Effusion" in TTAMAX_OVERLAY_FINDINGS
    assert "MCL" not in TTAMAX_OVERLAY_FINDINGS
    weights = overlay_weights_from_findings(TTAMAX_OVERLAY_FINDINGS, TTAMAX_OVERLAY_WEIGHT)
    assert not is_uniform_all_finding_overlay(weights, min_weight=0.5)
    assert overlay_finding_set(weights) == frozenset(TTAMAX_OVERLAY_FINDINGS)
    anchor = _table(11)
    overlay = _table(12)
    out = per_finding_rank_blend(anchor, overlay, weights)
    for name in FINDINGS:
        if name in TTAMAX_OVERLAY_FINDINGS:
            assert not np.allclose(out[name].to_numpy(), anchor[name].to_numpy())
        else:
            np.testing.assert_array_equal(out[name].to_numpy(), anchor[name].to_numpy())


def test_ttamax_kernel_is_not_ruled_out_clone():
    root = Path(__file__).resolve().parents[1]
    src = kernel_source_text(root / "kaggle_kernel_ttamax" / "main.py")
    assert "per_finding_rank_blend" in src
    assert "TTAMAX_OVERLAY_FINDINGS" in src
    assert "[tta-max]" in src
    assert "PF OA" in src
    assert "Effusion" in src
    assert "raptor_ft_cnv2b336.pt" not in src
    assert "raptor-knee-native384" not in src
    assert "focal_fold" not in src
    assert "not mill-RadImageNet" in src
    assert "_v41_selected = ['Medial OA', 'Synovitis', 'Contusion']" not in src
    assert "Baker's" in src


def test_r336ov_overlay_is_fail_closed_and_not_all_twelve():
    assert R336OV_OVERLAY_WEIGHT < 0.50
    assert R336OV_ENCODER == "dinov2-vitb14-r336"
    assert R336OV_OVERLAY_FINDINGS == ("PF OA", "Synovitis")
    assert is_ruled_out_r336ov_family(R336OV_OVERLAY_FINDINGS)
    assert is_ruled_out_r336ov_family(R336OV_OVERLAY_FINDINGS, weight=R336OV_OVERLAY_WEIGHT)
    assert not is_ruled_out_r336ov_family(R336OV_OVERLAY_FINDINGS, weight=0.15)
    assert "Effusion" not in R336OV_OVERLAY_FINDINGS
    assert "MCL" not in R336OV_OVERLAY_FINDINGS
    assert "Baker's" not in R336OV_OVERLAY_FINDINGS
    weights = overlay_weights_from_findings(R336OV_OVERLAY_FINDINGS, R336OV_OVERLAY_WEIGHT)
    assert not is_uniform_all_finding_overlay(weights, min_weight=0.5)
    assert overlay_finding_set(weights) == frozenset(R336OV_OVERLAY_FINDINGS)
    anchor = _table(13)
    overlay = _table(14)
    out = per_finding_rank_blend(anchor, overlay, weights)
    for name in FINDINGS:
        if name in R336OV_OVERLAY_FINDINGS:
            assert not np.allclose(out[name].to_numpy(), anchor[name].to_numpy())
        else:
            np.testing.assert_array_equal(out[name].to_numpy(), anchor[name].to_numpy())


def test_convnext_oof_k7_blend_moves_only_the_seven_findings():
    assert CONVNEXT_OOF_K7_WEIGHT == 0.15
    assert CONVNEXT_OOF_K7_WEIGHT < 0.50
    assert len(CONVNEXT_OOF_K7_FINDINGS) == 7
    weights = convnext_oof_k7_weights()
    assert overlay_finding_set(weights) == frozenset(CONVNEXT_OOF_K7_FINDINGS)
    assert not is_uniform_all_finding_overlay(weights, min_weight=0.5)
    assert not is_ruled_out_r336ov_family(CONVNEXT_OOF_K7_FINDINGS, weight=CONVNEXT_OOF_K7_WEIGHT)
    assert not is_ruled_out_dino3s_replace_family(CONVNEXT_OOF_K7_FINDINGS, weight=CONVNEXT_OOF_K7_WEIGHT)
    assert not is_ruled_out_public_native384_family(CONVNEXT_OOF_K7_FINDINGS)
    assert not is_ruled_out_jointline_mill_radimagenet_family(CONVNEXT_OOF_K7_FINDINGS)
    anchor = _table(13)
    overlay = _table(14)
    out = fail_closed_rank_blend(anchor, overlay, weights)
    for name in FINDINGS:
        if name in CONVNEXT_OOF_K7_FINDINGS:
            assert not np.allclose(out[name].to_numpy(), anchor[name].to_numpy())
        else:
            np.testing.assert_array_equal(out[name].to_numpy(), anchor[name].to_numpy())


def test_unequal_k7_weights_match_the_517_search_and_beat_uniform_045():
    root = Path(__file__).resolve().parents[1]
    measured = json.loads(
        (root / "artifacts" / "convnext_oof870" / "k7_unequal_strict.json").read_text(encoding="utf-8")
    )
    weights = convnext_oof_unequal_weights()
    assert weights == {name: float(measured["weights"][name]) for name in FINDINGS}
    assert all(value < 0.50 for value in weights.values())
    assert weights["Lateral Meniscus"] == 0.0
    assert weights["PF OA"] == 0.0
    assert weights["Lateral OA"] == 0.0
    assert weights["Contusion"] == 0.0
    assert weights["Fracture"] == 0.0
    assert measured["all_kept_lo_gt_0"] is True
    assert float(measured["macro_lo"]) > 0.003
    for name, row in measured["per_finding"].items():
        assert weights[name] == 0.49
        assert row["lo"] > 0.0
    pred = pd.read_csv(root / "artifacts" / "convnext_oof870" / "convnext_oof870.csv")
    mill = pd.read_csv(root / "data" / "mill_v10_holdout870.csv")
    train = pd.read_csv(root / "data" / "train.csv")
    v4 = pd.read_csv(root / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv")
    from rsna_knee.metrics import macro_roc_auc
    from rsna_knee.mill_holdout import holdout_ids_and_y

    holdout, y_hold = holdout_ids_and_y(train, v4)
    y_map = {uid: y_hold[i] for i, uid in enumerate(holdout)}
    ids = pred[ID_COLUMN].astype(str).tolist()
    y = np.stack([y_map[str(uid)] for uid in ids])
    order = pred[ID_COLUMN].astype(str)
    mill_v = mill.set_index(ID_COLUMN).loc[order, list(FINDINGS)].to_numpy(np.float64)
    new_v = pred.set_index(ID_COLUMN).loc[order, list(FINDINGS)].to_numpy(np.float64)
    stub = [str(i) for i in range(len(ids))]

    def _delta(blend_weights: dict[str, float]) -> float:
        left = pd.DataFrame(mill_v, columns=list(FINDINGS))
        left.insert(0, ID_COLUMN, stub)
        right = pd.DataFrame(new_v, columns=list(FINDINGS))
        right.insert(0, ID_COLUMN, stub)
        mixed = per_finding_rank_blend(
            left[list(SUBMISSION_COLUMNS)],
            right[list(SUBMISSION_COLUMNS)],
            blend_weights,
        )
        return float(
            macro_roc_auc(y, mixed[list(FINDINGS)].to_numpy(np.float64))
            - macro_roc_auc(y, mill_v)
        )

    assert len(ids) == 517
    assert _delta(weights) > _delta(convnext_oof_k7_hi_weights())
    anchor = _table(13)
    overlay = _table(14)
    out = fail_closed_rank_blend(anchor, overlay, weights)
    for name in FINDINGS:
        if weights[name] > 0.0:
            assert not np.allclose(out[name].to_numpy(), anchor[name].to_numpy())
        else:
            np.testing.assert_array_equal(out[name].to_numpy(), anchor[name].to_numpy())


def test_k7_hi_weight_beats_uniform_015_on_the_pinned_oof_table():
    root = Path(__file__).resolve().parents[1]
    pred = pd.read_csv(root / "artifacts" / "convnext_oof870" / "convnext_oof870.csv")
    mill = pd.read_csv(root / "data" / "mill_v10_holdout870.csv")
    train = pd.read_csv(root / "data" / "train.csv")
    v4 = pd.read_csv(root / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv")
    from rsna_knee.metrics import macro_roc_auc
    from rsna_knee.mill_holdout import holdout_ids_and_y

    holdout, y_hold = holdout_ids_and_y(train, v4)
    y_map = {uid: y_hold[i] for i, uid in enumerate(holdout)}
    ids = pred[ID_COLUMN].astype(str).tolist()
    y = np.stack([y_map[uid] for uid in ids])
    order = pred[ID_COLUMN].astype(str)
    mill_v = mill.set_index(ID_COLUMN).loc[order, list(FINDINGS)].to_numpy(np.float64)
    new_v = pred.set_index(ID_COLUMN).loc[order, list(FINDINGS)].to_numpy(np.float64)
    stub = [str(i) for i in range(len(ids))]

    def _delta(weights: dict[str, float]) -> float:
        frame = pd.DataFrame(mill_v, columns=list(FINDINGS))
        frame.insert(0, ID_COLUMN, stub)
        other = pd.DataFrame(new_v, columns=list(FINDINGS))
        other.insert(0, ID_COLUMN, stub)
        mixed = per_finding_rank_blend(
            frame[list(SUBMISSION_COLUMNS)],
            other[list(SUBMISSION_COLUMNS)],
            weights,
        )
        return float(
            macro_roc_auc(y, mixed[list(FINDINGS)].to_numpy(np.float64))
            - macro_roc_auc(y, mill_v)
        )

    base = _delta(convnext_oof_k7_weights())
    hi = _delta(convnext_oof_k7_hi_weights())
    assert len(ids) == 517
    assert hi > base + 0.004
    assert base > 0.003


def test_k7_hi_weight_stays_under_the_fail_closed_wall_and_moves_the_seven():
    hi = convnext_oof_k7_hi_weights()
    assert CONVNEXT_OOF_K7_HI_WEIGHT == 0.45
    assert all(hi[name] == 0.45 for name in CONVNEXT_OOF_K7_FINDINGS)
    assert overlay_finding_set(hi) == frozenset(CONVNEXT_OOF_K7_FINDINGS)
    with pytest.raises(FusionError):
        convnext_oof_k7_weights(0.50)
    anchor = _table(13)
    overlay = _table(14)
    out = fail_closed_rank_blend(anchor, overlay, hi)
    for name in FINDINGS:
        if name in CONVNEXT_OOF_K7_FINDINGS:
            assert not np.allclose(out[name].to_numpy(), anchor[name].to_numpy())
        else:
            np.testing.assert_array_equal(out[name].to_numpy(), anchor[name].to_numpy())


def test_k7_three_row_log_changes_only_the_finding_whose_order_moves():
    """n=3 re-rank equals mill wherever the blend keeps mill's order.

    The Save Version line overlay_columns=['Effusion'] is that order check.
    All seven weights are 0.15. Effusion is the column whose tie breaks.
    """
    from rsna_knee.convnext_overlay_kernel import blend_tables

    weights = convnext_oof_k7_weights()
    assert overlay_finding_set(weights) == frozenset(CONVNEXT_OOF_K7_FINDINGS)
    assert all(weights[name] == 0.15 for name in CONVNEXT_OOF_K7_FINDINGS)
    ids = ["row-a", "row-b", "row-c"]
    mill_vals = {
        "ACL": [1 / 3, 2 / 3, 1.0],
        "MCL": [1 / 3, 1.0, 2 / 3],
        "Medial Meniscus": [2 / 3, 1.0, 1 / 3],
        "Lateral Meniscus": [1 / 3, 1.0, 2 / 3],
        "Medial OA": [2 / 3, 1.0, 1 / 3],
        "Lateral OA": [1 / 3, 2 / 3, 1.0],
        "PF OA": [1 / 3, 2 / 3, 1.0],
        "Effusion": [0.5, 1.0, 0.5],
        "Synovitis": [1 / 3, 1.0, 2 / 3],
        "Baker's": [1.0, 2 / 3, 1 / 3],
        "Contusion": [0.5, 1.0, 0.5],
        "Fracture": [1 / 3, 1.0, 2 / 3],
    }
    # Same order as mill on every weighted finding except Effusion, where the
    # reader ranks row-a above row-c and breaks mill's 0.5/0.5 tie.
    reader_vals = {name: list(vals) for name, vals in mill_vals.items()}
    reader_vals["Effusion"] = [0.52, 0.30, 0.36]
    mill = _rows(ids, mill_vals)
    reader = _rows(ids, reader_vals)
    out = blend_tables(mill, reader, weights)
    changed = [name for name in CONVNEXT_OOF_K7_FINDINGS if not out[name].equals(mill[name])]
    assert changed == ["Effusion"]
    assert list(out["Effusion"]) == [2 / 3, 1.0, 1 / 3]
    for name in FINDINGS:
        if name in CONVNEXT_OOF_K7_FINDINGS:
            continue
        np.testing.assert_array_equal(out[name].to_numpy(), mill[name].to_numpy())
    # A strict 3-row order is not flipped by weight 0.15. The log line is a
    # tie-break, not evidence that the other six weights were zero.
    reader_vals["ACL"] = [0.9, 0.1, 0.2]
    held = blend_tables(mill, _rows(ids, reader_vals), weights)
    assert held["ACL"].equals(mill["ACL"])
    assert not held["Effusion"].equals(mill["Effusion"])


def _rows(ids: list[str], columns: dict[str, list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(columns, index=range(len(ids)))
    frame.insert(0, ID_COLUMN, ids)
    return frame[list(SUBMISSION_COLUMNS)]


def test_r336ov_kernel_is_not_ruled_out_clone():
    root = Path(__file__).resolve().parents[1]
    src = kernel_source_text(root / "kaggle_kernel_r336ov" / "main.py")
    assert "GOLD_ONLY = True" not in src
    assert "r336_best.pt" in src
    assert "[r336ov]" in src
    assert "R336OV_OVERLAY_FINDINGS" in src
    assert "PF OA" in src
    assert "Synovitis" in src
    assert "per_finding_rank_blend" in src
    assert "raptor_ft_cnv2b336.pt" not in src
    assert "raptor-knee-native384" not in src
    assert "focal_fold" not in src
    assert "_v41_selected = ['Medial OA', 'Synovitis', 'Contusion']" not in src
    assert "Baker's" in src


def test_goldfold_kernel_is_not_ruled_out_clone():
    root = Path(__file__).resolve().parents[1]
    src = kernel_source_text(root / "kaggle_kernel_goldfold" / "main.py")
    assert GOLDFOLD_ENCODER == "dinov2-vitb14-r336"
    assert GOLDFOLD_N_GOLD == 58
    assert GOLDFOLD_OVERLAY_WEIGHT == 0.0
    assert "GOLD_ONLY = True" in src
    assert "r336_best.pt" in src
    assert "[goldfold]" in src
    assert "DO NOT SUBMIT" in src
    assert "Dinov2Config" in src
    assert "pytorch_model.bin" in src
    assert "not overlay" in src
    assert "raptor_ft_cnv2b336.pt" not in src
    assert "raptor-knee-native384" not in src
    assert "focal_fold" not in src
    assert "_v41_selected = ['Medial OA', 'Synovitis', 'Contusion']" not in src
    assert "Baker's" in src


def test_dino3s_kernel_is_not_ruled_out_clone():
    root = Path(__file__).resolve().parents[1]
    src = kernel_source_text(root / "kaggle_kernel_dino3s" / "main.py")
    assert "per_finding_rank_blend" in src
    assert "DINO3S_REPLACE_FINDINGS" in src
    assert "dinov3-vits16plus" in src
    assert "[dino3s]" in src
    assert "raptor_ft_cnv2b336.pt" not in src
    assert "raptor-knee-native384" not in src
    assert "JOINTLINE_CROP_MM" not in src or "not mill-RadImageNet" in src
    assert "_v41_selected = ['Medial OA', 'Synovitis', 'Contusion']" not in src
    assert "focal_fold" in src
    assert "Baker's" in src


def test_jointline_kernel_is_not_native384_or_cnn_tri_clone():
    root = Path(__file__).resolve().parents[1]
    src = kernel_source_text(root / "kaggle_kernel_jointline" / "main.py")
    assert "per_finding_rank_blend" in src
    assert "JOINTLINE_OVERLAY_FINDINGS" in src
    assert "JOINTLINE_CROP_MM" in src
    assert "[jointline]" in src
    assert "90" in src
    assert "raptor_ft_cnv2b336.pt" not in src
    assert "native384" not in src.lower() or "not native384" in src.lower()
    assert "_v41_selected = ['Medial OA', 'Synovitis', 'Contusion']" not in src
    assert "raptor-knee-native384" not in src


def test_fail_closed_effusion_synovitis_blends_only_those_columns():
    anchor = _table(21)
    overlay = _table(22)
    weights = overlay_weights_from_findings(("Effusion", "Synovitis"), 0.15)
    out = fail_closed_rank_blend(anchor, overlay, weights)
    assert list(out.columns) == list(SUBMISSION_COLUMNS)
    assert "Baker's" in out.columns
    for name in FINDINGS:
        if name in {"Effusion", "Synovitis"}:
            assert not np.allclose(out[name].to_numpy(), anchor[name].to_numpy())
        else:
            np.testing.assert_array_equal(out[name].to_numpy(), anchor[name].to_numpy())


def test_fail_closed_returns_anchor_on_dummy_nan_flat_and_ruled_out_pairs():
    anchor = _table(23)
    dummy = anchor.copy()
    dummy["Effusion"] = 0.5
    dummy["Synovitis"] = 1.0
    weights = overlay_weights_from_findings(("Effusion", "Synovitis"), 0.15)
    out = fail_closed_rank_blend(anchor, dummy, weights)
    pd.testing.assert_frame_equal(out, anchor)

    missing = overlay = _table(24)
    missing = overlay.copy()
    missing.loc[0, "Synovitis"] = np.nan
    out = fail_closed_rank_blend(anchor, missing, weights)
    pd.testing.assert_frame_equal(out, anchor)

    flat = _table(25)
    flat["Effusion"] = 0.2
    out = fail_closed_rank_blend(anchor, flat, weights)
    pd.testing.assert_frame_equal(out, anchor)

    ruled = overlay_weights_from_findings(("PF OA", "Synovitis"), 0.20)
    out = fail_closed_rank_blend(anchor, _table(26), ruled)
    pd.testing.assert_frame_equal(out, anchor)

    heavy = overlay_weights_from_findings(("Effusion",), 0.50)
    out = fail_closed_rank_blend(anchor, _table(27), heavy)
    pd.testing.assert_frame_equal(out, anchor)

    short = anchor.iloc[:-1].reset_index(drop=True)
    out = fail_closed_rank_blend(anchor, short, weights)
    pd.testing.assert_frame_equal(out, anchor)


def test_logfix_kernel_uses_shipped_fusion_not_cnn_tri_pattern():
    root = Path(__file__).resolve().parents[1]
    src = kernel_source_text(root / "kaggle_kernel_logfix" / "main.py")
    assert "per_finding_rank_blend" in src
    assert "LOGFIX_OVERLAY_FINDINGS" in src
    assert "from fusion import" in src
    assert "uniform overlay weight >= 0.50 on all 12" not in src or "CNN-tri" in src
    assert "_v41_labs" not in src
    assert "raptor_ft_cnv2b336.pt" not in src
    assert "_v41_selected = ['Medial OA', 'Synovitis', 'Contusion']" not in src
    assert "LOGFIX_OVERLAY_WEIGHT" in src
    assert "write_text(_FUSION_PY" in src
