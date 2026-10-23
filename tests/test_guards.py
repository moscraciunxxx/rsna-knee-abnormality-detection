"""Drive shipped dummy-rank and kernel-source guards on real files."""

from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.guards import (
    header_is_official,
    is_dummy_rank_trap,
    kernel_discovers_competitions_mount,
    kernel_has_t4_gpu_probe,
    kernel_keeps_bakers_apostrophe,
    kernel_reads_test_report_column,
    kernel_source_text,
)
from rsna_knee.infer import _find_dir_with_test_csv
from rsna_knee.schema import SUBMISSION_COLUMNS
from rsna_knee.submission import build_submission

ROOT = Path(__file__).resolve().parents[1]
KERNELS = [
    ROOT / "kaggle_kernel_blend_script" / "main.py",
    ROOT / "kaggle_kernel_hs_v41" / "main.py",
    ROOT / "kaggle_kernel_cnn_tri" / "main.py",
    ROOT / "kaggle_kernel_logfix" / "main.py",
    ROOT / "kaggle_kernel_jointline" / "main.py",
    ROOT / "kaggle_kernel_dino3s" / "main.py",
    ROOT / "kaggle_kernel_ttamax" / "main.py",
    ROOT / "kaggle_kernel_goldfold" / "main.py",
    ROOT / "kaggle_kernel_r336ov" / "main.py",
    ROOT / "kaggle_kernel_v4head" / "main.py",
    ROOT / "kaggle_kernel_mednet" / "main.py",
    ROOT / "kaggle_kernel_pfoa_syn" / "main.py",
    ROOT / "kaggle_kernel_attn25d" / "main.py",
    ROOT / "kaggle_kernel_mill_holdout" / "main.py",
    ROOT / "kaggle_kernel_dense12" / "main.py",
    ROOT / "kaggle_kernel_vitb870" / "main.py",
    ROOT / "kaggle_kernel_coatnet870_gate" / "main.py",
]


def test_dummy_rank_trap_flags_p100_pattern():
    dummy = np.array([[0.0, 0.5, 1.0] * 4], dtype=float)
    assert dummy.shape == (1, 12)
    assert is_dummy_rank_trap(dummy)


def test_dummy_rank_trap_allows_fractional_ranks():
    real = np.array([[1 / 3, 2 / 3, 0.5] * 4], dtype=float)
    assert not is_dummy_rank_trap(real)


def test_build_submission_fractional_ranks_are_not_dummy():
    table = build_submission(
        ["s1", "s2", "s3"],
        np.array(
            [
                [1 / 3] * 12,
                [2 / 3] * 12,
                [0.5] * 12,
            ]
        ),
    )
    values = table[list(SUBMISSION_COLUMNS[1:])].to_numpy()
    assert header_is_official(table.columns)
    assert "Baker's" in table.columns
    assert not is_dummy_rank_trap(values)


def test_scoring_kernels_have_t4_probe_bakers_and_competitions_mount():
    for path in KERNELS:
        src = kernel_source_text(path)
        assert kernel_has_t4_gpu_probe(src), path
        assert kernel_discovers_competitions_mount(src), path
        assert kernel_keeps_bakers_apostrophe(src), path
        assert not kernel_reads_test_report_column(src), path
        assert "1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191" not in src


def test_cnn_tri_kernel_is_convnext_effnet_not_v10_clone():
    src = kernel_source_text(ROOT / "kaggle_kernel_cnn_tri" / "main.py")
    assert "raptor_ft_cnv2b336.pt" in src
    assert "raptor_ft_effv2l480.pt" in src
    assert "convnextv2_base.fcmae_ft_in22k_in1k_384" in src
    assert "tf_efficientnetv2_l.in21k_ft_in1k" in src
    assert "CNN-tri" in src
    assert "raptor_ft_coatnet_v10_full.pt" not in src


def test_discovered_study_count_matches_infer_and_convnext_clock_blocks_large_n(tmp_path: Path):
    import base64
    import importlib.util
    import json
    import zlib

    from rsna_knee.infer import discover_study_ids
    from rsna_knee.scoring_time import N_PLAN_CACHE, convnext_overlay_clock_allowed
    from rsna_knee.study_count import discovered_study_ids

    study = tmp_path / "test_series" / "9.9.9.not.a.public.test.id"
    study.mkdir(parents=True)
    (study / "slice.dcm").write_bytes(b"dcm")
    (tmp_path / "test.csv").write_text(
        "StudyInstanceUID\n9.9.9.not.a.public.test.id\ncsv-only-study\n",
        encoding="utf-8",
    )
    assert discovered_study_ids(tmp_path) == discover_study_ids(tmp_path)
    assert convnext_overlay_clock_allowed(N_PLAN_CACHE)
    assert convnext_overlay_clock_allowed(0) is False
    n = 1
    while convnext_overlay_clock_allowed(n):
        n += 1
    assert n > N_PLAN_CACHE

    builder_path = ROOT / "scripts" / "build_convnext_eff_syn_overlay_kernel.py"
    spec = importlib.util.spec_from_file_location("build_cnxt_ov", builder_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    built = module.build_kernel(dst=ROOT / "kaggle_kernel_convnext_eff_syn_overlay")
    text = (built / "main.py").read_text(encoding="utf-8")
    assert "DO NOT" in text or "55779066" in text
    assert "1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191" not in text
    ns: dict = {"__name__": "kernel_blob"}
    exec(compile(text, str(built / "main.py"), "exec"), ns)
    blobs = ns["BLOBS"]
    for name in ("fusion.py", "scoring_time.py", "study_count.py", "convnext_overlay_kernel.py"):
        raw = zlib.decompress(base64.b64decode(blobs[name]))
        assert raw == (ROOT / "rsna_knee" / name).read_bytes()
    kernel = zlib.decompress(base64.b64decode(blobs["convnext_overlay_kernel.py"])).decode("utf-8")
    assert "Baker's" in kernel
    assert "Effusion" in kernel and "Synovitis" in kernel
    assert "0.15" in kernel
    assert "_holdout_comp" not in kernel
    assert "['Report']" not in kernel
    assert '["Report"]' not in kernel
    assert "/kaggle/input/competitions/rsna-knee-abnormality-detection" in kernel
    assert "nvidia-smi" in kernel and "CUDA_VISIBLE_DEVICES" in kernel
    meta = json.loads((built / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["enable_internet"] is False
    assert meta["enable_gpu"] is True
    assert "goodpjw2008/rsna-knee-2-5d-convnext-reader" in meta["dataset_sources"]
    assert meta["competition_sources"] == ["rsna-knee-abnormality-detection"]

    bad = tmp_path / "bad_spec.json"
    bad.write_text(json.dumps({"decision": "stop", "findings": [], "weight": 0}), encoding="utf-8")
    try:
        module.build_kernel(spec_path=bad, dst=tmp_path / "should_not")
    except SystemExit as exc:
        assert "decision" in str(exc)
    else:
        raise AssertionError("bad spec must refuse")


def test_densenet_holdout_artifact_stops_and_is_not_submitted():
    from rsna_knee.mill_holdout import decide_private_overlay, holdout_ids_and_y, join_mill_holdout
    from rsna_knee.schema import ID_COLUMN

    import pandas as pd

    root_note = (ROOT / "artifacts" / "densenet121_head" / "DO_NOT_SUBMIT_HOLDOUT.txt").read_text(encoding="utf-8")
    protocol = (ROOT / "artifacts" / "densenet121_head" / "protocol_log.txt").read_text(encoding="utf-8")
    assert "not the hidden test" in root_note
    assert "55779066" in root_note
    assert "DO NOT SUBMIT" in protocol
    assert "gold_in_fit=0" in protocol
    assert "abort=False" in protocol
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv", dtype={ID_COLUMN: str})
    ids, y = holdout_ids_and_y(train, v4)
    mill = pd.read_csv(ROOT / "data" / "mill_v10_holdout870.csv", dtype={ID_COLUMN: str})
    new = pd.read_csv(ROOT / "artifacts" / "densenet121_head" / "densenet_holdout870.csv", dtype={ID_COLUMN: str})
    decision, weights = decide_private_overlay(
        join_mill_holdout(ids, mill, y).per_auc,
        join_mill_holdout(ids, new, y).per_auc,
    )
    assert decision == "stop"
    assert weights is None


def test_densenet_head_split_matches_mill_holdout_and_is_not_a_submit():
    import importlib.util
    import json

    import pandas as pd

    from rsna_knee.densenet_train import densenet_id_split, fit_linear_head, gold_gate
    from rsna_knee.mill_holdout import holdout_ids_and_y
    from rsna_knee.schema import ID_COLUMN

    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv", dtype={ID_COLUMN: str})
    fit_ids, hold_ids, gold_ids = densenet_id_split(train, v4)
    expected, _y = holdout_ids_and_y(train, v4)
    assert hold_ids == expected
    assert len(gold_ids) == 58
    assert len(hold_ids) == 870
    assert set(fit_ids).isdisjoint(hold_ids)
    assert set(fit_ids).isdisjoint(gold_ids)
    assert set(hold_ids).isdisjoint(gold_ids)
    rng = np.random.default_rng(0)
    x_fit = rng.normal(size=(32, 8))
    y_fit = rng.random((32, 12))
    w_fit = np.ones_like(y_fit)
    scores = fit_linear_head(x_fit, y_fit, w_fit, rng.normal(size=(6, 8)), epochs=2)
    assert scores.shape == (6, 12)
    assert np.isfinite(scores).all()
    y_gold = np.tile(np.array([0.0, 1.0], dtype=np.float64), (6, 6))
    _macro, abort = gold_gate(y_gold, np.tile(np.linspace(0.1, 0.9, 12), (6, 1)))
    assert abort is False

    builder_path = ROOT / "scripts" / "build_densenet121_head_kernel.py"
    spec = importlib.util.spec_from_file_location("build_dn121_head", builder_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    built = module.build_kernel()
    text = (built / "main.py").read_text(encoding="utf-8")
    meta = json.loads((built / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert "DO NOT SUBMIT" in text
    assert "gold_in_fit" in text
    assert "densenet_holdout870.csv" in text
    assert "['Report']" not in text and '["Report"]' not in text
    assert "competitions submit" not in text
    assert meta["enable_internet"] is False
    assert "sjudal/radimagenet-densenet121" in meta["dataset_sources"]
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]


def test_densenet_probe_excludes_gold_and_aborts_a_slow_encode():
    import importlib.util
    import json

    import pandas as pd

    from rsna_knee.densenet_probe import encode_probe_aborts, probe_study_ids, projected_encode_seconds
    from rsna_knee.schema import ID_COLUMN
    from rsna_knee.v4_split import expert_gold_ids, should_abort_gold_overfit

    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    gold = set(expert_gold_ids(train))
    chosen = probe_study_ids(train)
    assert len(chosen) == 8
    assert set(chosen).isdisjoint(gold)
    slow = projected_encode_seconds(144.0, 144)
    fast = projected_encode_seconds(10.0, 144)
    assert encode_probe_aborts(144.0, 144) is True
    assert slow > 2.8 * 3600
    assert encode_probe_aborts(10.0, 144) is False
    assert fast < 2.8 * 3600
    assert should_abort_gold_overfit(0.95) is True
    assert should_abort_gold_overfit(0.93) is False

    builder_path = ROOT / "scripts" / "build_densenet121_probe_kernel.py"
    spec = importlib.util.spec_from_file_location("build_dn121", builder_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    built = module.build_kernel()
    text = (built / "main.py").read_text(encoding="utf-8")
    meta = json.loads((built / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert "DO NOT SUBMIT" in text
    assert "Baker's" in text
    assert "['Report']" not in text and '["Report"]' not in text
    assert "gold_in_probe" in text
    assert "encode_probe_aborts" in text
    assert ".astype(np.float32)" in text
    assert ".float()" in text
    assert meta["enable_internet"] is False
    assert meta["enable_gpu"] is True
    assert "sjudal/radimagenet-densenet121" in meta["dataset_sources"]
    assert "competitions submit" not in text


def test_radimagenet_densenet121_notop_loads_a_new_twelve_way_head():
    import torch

    from rsna_knee.radimagenet_densenet import load_radimagenet_densenet121

    path = ROOT / ".cache" / "radimagenet" / "RadImageNet-DenseNet121_notop.h5"
    model, report = load_radimagenet_densenet121(path)
    assert report["unexpected"] == []
    assert report["n_loaded"] > 400
    assert report["classifier_out"] == 12
    assert report["features_frozen"] is True
    assert report["head_trainable"] is True
    allowed = {
        key
        for key in report["missing"]
        if key.endswith("num_batches_tracked") or key.startswith("classifier.")
    }
    assert set(report["missing"]) == allowed
    with torch.no_grad():
        features = model.features(torch.zeros(1, 3, 224, 224))
    assert tuple(features.shape) == (1, 1024, 7, 7)
    assert torch.isfinite(features).all()


def test_find_dir_with_test_csv_nested_competitions_mount(tmp_path: Path):
    nested = tmp_path / "competitions" / "rsna-knee-abnormality-detection"
    nested.mkdir(parents=True)
    pd.DataFrame({"StudyInstanceUID": ["a"]}).to_csv(nested / "test.csv", index=False)
    found = _find_dir_with_test_csv(tmp_path)
    assert found == nested


def test_convnext_holdout_licenses_effusion_synovitis_and_is_not_a_submit():
    import json

    from rsna_knee.mill_holdout import decide_private_overlay, holdout_ids_and_y, join_mill_holdout
    from rsna_knee.schema import ID_COLUMN

    note = (ROOT / "artifacts" / "convnext_mil870" / "DO_NOT_SUBMIT_HOLDOUT.txt").read_text(encoding="utf-8")
    kernel = (ROOT / "kaggle_kernel_convnext_mil870_gate" / "main.py").read_text(encoding="utf-8")
    assert "not the hidden test" in note
    assert "55779066" in note
    assert "DO NOT SUBMIT" in kernel
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv", dtype={ID_COLUMN: str})
    ids, y = holdout_ids_and_y(train, v4)
    mill = pd.read_csv(ROOT / "data" / "mill_v10_holdout870.csv", dtype={ID_COLUMN: str})
    new = pd.read_csv(ROOT / "artifacts" / "convnext_mil870" / "convnext_holdout870.csv", dtype={ID_COLUMN: str})
    decision, weights = decide_private_overlay(
        join_mill_holdout(ids, mill, y).per_auc,
        join_mill_holdout(ids, new, y).per_auc,
    )
    assert decision == "overlay"
    assert weights is not None
    licensed = {name for name, weight in weights.items() if weight > 0}
    assert licensed == {"Effusion", "Synovitis"}
    assert weights["Effusion"] == 0.15
    assert weights["Synovitis"] == 0.15
    assert weights["PF OA"] == 0.0
    spec = json.loads((ROOT / "artifacts" / "convnext_mil870" / "overlay_spec.json").read_text(encoding="utf-8"))
    assert spec["decision"] == "overlay"
    assert spec["keep"] == "55779066"
    assert spec["do_not_submit_holdout_ids"] is True
    assert spec["source"] == "convnext-mil870-gate"
    assert set(spec["findings"]) == {"Effusion", "Synovitis"}
    assert spec["weight"] == 0.15


def test_convnext_oof_folds_match_author_split_and_drop_folds_3_and_4():
    import json

    from rsna_knee.convnext_oof import author_v0_folds, oof_ids_by_fold, report_group
    from rsna_knee.mill_holdout import holdout_ids_and_y
    from rsna_knee.schema import FINDINGS, ID_COLUMN

    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv", dtype={ID_COLUMN: str})
    folds = author_v0_folds(train)
    gold = train[list(FINDINGS)].notna().all(axis=1)
    assert int((folds == -1).sum()) == 58
    assert set(folds.index[folds == -1]) == set(train.loc[gold, ID_COLUMN].astype(str))
    weak = folds[folds >= 0]
    assert set(weak.unique().tolist()) == {0, 1, 2, 3, 4}
    groups = train.assign(_g=train["Report"].map(report_group))
    for group, rows in groups.groupby("_g"):
        got = set(int(folds.loc[uid]) for uid in rows[ID_COLUMN].astype(str) if int(folds.loc[uid]) >= 0)
        assert len(got) <= 1, group
    ids, _y = holdout_ids_and_y(train, v4)
    by_fold = oof_ids_by_fold(folds, ids)
    oof = [uid for fold_id in (0, 1, 2) for uid in by_fold[fold_id]]
    assert 300 < len(oof) < 870
    assert set(oof).isdisjoint(set(folds.index[folds == -1]))
    assert all(int(folds.loc[uid]) == fold_id for fold_id, fold_ids in by_fold.items() for uid in fold_ids)
    assert not any(int(folds.loc[uid]) in (3, 4) for uid in oof)
    kernel = (ROOT / "kaggle_kernel_convnext_oof870" / "main.py").read_text(encoding="utf-8")
    meta = (ROOT / "kaggle_kernel_convnext_oof870" / "kernel-metadata.json").read_text(encoding="utf-8")
    code = kernel.split('"""', 2)[-1]
    assert "GroupKFold" not in code
    assert "Report" not in code
    assert "PINNED_HOLDOUT_FOLDS" in code
    assert "DO NOT SUBMIT" in kernel
    assert "submission.csv" in kernel
    assert '"enable_internet": false' in meta
    assert "rsna-knee-convnext-oof870" in meta
    start = kernel.index("json.loads(r'''") + len("json.loads(r'''")
    end = kernel.index("''')", start)
    pinned = json.loads(kernel[start:end])
    assert len(pinned) == 870
    assert {uid: int(folds.loc[uid]) for uid in ids} == {k: int(v) for k, v in pinned.items()}
    assert sum(int(v) == 0 for v in pinned.values()) == 174
    assert sum(int(v) == 1 for v in pinned.values()) == 183
    assert sum(int(v) == 2 for v in pinned.values()) == 160
    spec = json.loads((ROOT / "artifacts" / "convnext_oof870" / "overlay_spec.json").read_text(encoding="utf-8"))
    boot = json.loads((ROOT / "artifacts" / "convnext_oof870" / "bootstrap_ci.json").read_text(encoding="utf-8"))
    refused = json.loads((ROOT / "artifacts" / "convnext_oof870" / "v1_not_a_license.json").read_text(encoding="utf-8"))
    assert spec["decision"] == "overlay"
    assert spec["findings"] == ["ACL", "Effusion"]
    assert spec["weight"] == 0.15
    assert spec["do_not_submit_holdout_ids"] is True
    assert spec["test_submit"] is False
    assert spec["keep"] == "55779066"
    assert boot["bar_macro_lo_ge_0.003"] is False
    assert boot["do_not_submit"] is True
    assert boot["macro_delta_ci95"]["lo"] < 0.003
    assert refused["decision"] == "hold"
    assert refused["fold_id_agree"] == "65 of 513"
    k7 = json.loads((ROOT / "artifacts" / "convnext_oof870" / "bootstrap_k7.json").read_text(encoding="utf-8"))
    assert k7["do_not_submit"] is True
    assert k7["keep_anchor"] == "55779066"
    assert k7["seven"]["clears_0.003"] is True
    assert k7["seven"]["macro_lo"] >= 0.003
    assert len(k7["seven"]["findings"]) == 7
    assert all(row["lo_gt_0"] for row in k7["seven"]["per_finding"].values())


def test_k7_overlay_package_fail_closes_and_stays_unpushed(tmp_path: Path):
    import base64
    import importlib.util
    import json
    import zlib

    from rsna_knee.convnext_k7_overlay_kernel import (
        HEADER,
        control_score_allows_push,
    )
    from rsna_knee.convnext_overlay_kernel import refuse_over_clock
    from rsna_knee.schema import SUBMISSION_COLUMNS
    from rsna_knee.scoring_time import (
        convnext_overlay_clock_allowed,
        convnext_overlay_estimate_s,
    )

    assert HEADER == ",".join(SUBMISSION_COLUMNS)
    assert "Baker's" in HEADER
    assert control_score_allows_push(None) is False
    assert control_score_allows_push(0.935) is False
    assert control_score_allows_push(0.936) is True
    assert control_score_allows_push(0.937) is True
    last_ok = 1
    while convnext_overlay_clock_allowed(last_ok + 1):
        last_ok += 1
    assert convnext_overlay_estimate_s(last_ok) < 8.75 * 3600
    assert convnext_overlay_estimate_s(last_ok + 1) >= 8.75 * 3600
    doomed = tmp_path / "submission.csv"
    doomed.write_text("not-a-header\n", encoding="utf-8")
    try:
        refuse_over_clock(last_ok + 1, doomed)
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("over-clock discovery must exit 2")
    assert not doomed.is_file()

    builder_path = ROOT / "scripts" / "build_convnext_k7_overlay_kernel.py"
    spec = importlib.util.spec_from_file_location("build_k7_ov", builder_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    built = module.build_kernel(dst=ROOT / "kaggle_kernel_convnext_k7_overlay")
    text = (built / "main.py").read_text(encoding="utf-8")
    assert "convnext_k7_overlay_kernel" in text
    assert "55779066" in text
    public = (ROOT / "data" / "test.csv").read_text(encoding="utf-8")
    for line in public.splitlines()[1:]:
        uid = line.split(",")[0].strip()
        if uid:
            assert uid not in text
    ns: dict = {"__name__": "k7_blob"}
    exec(compile(text, str(built / "main.py"), "exec"), ns)
    blobs = ns["BLOBS"]
    for name in (
        "fusion.py",
        "scoring_time.py",
        "study_count.py",
        "convnext_overlay_kernel.py",
        "convnext_k7_overlay_kernel.py",
    ):
        raw = zlib.decompress(base64.b64decode(blobs[name]))
        assert raw == (ROOT / "rsna_knee" / name).read_bytes()
    entry = zlib.decompress(base64.b64decode(blobs["convnext_k7_overlay_kernel.py"])).decode("utf-8")
    runner = zlib.decompress(base64.b64decode(blobs["convnext_overlay_kernel.py"])).decode("utf-8")
    mill = zlib.decompress(base64.b64decode(blobs["_mill_v10.py"])).decode("utf-8")
    for name in ("ACL", "MCL", "Medial Meniscus", "Medial OA", "Effusion", "Synovitis", "Baker's"):
        assert name in entry or name in (ROOT / "rsna_knee" / "fusion.py").read_text(encoding="utf-8")
    assert "convnext_oof_k7_weights" in entry
    assert "run_overlay" in entry
    assert "fail_closed_rank_blend" in runner
    assert "discovered_study_ids" in runner
    assert "refuse_over_clock" in runner
    assert "walk('test_series')" in mill
    assert "Report" not in entry
    assert "['Report']" not in runner
    assert '["Report"]' not in runner
    meta = json.loads((built / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["enable_internet"] is False
    assert meta["enable_gpu"] is True
    assert meta["id"] == "vitaliecervinschi/rsna-knee-convnext-k7-ov"
    assert "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461" in meta["docker_image"]
    assert "2757e0c7" not in meta["docker_image"]
    assert meta["kernel_sources"] == ["romantamrazov/rsna-knee-dinosaur-v5-train"]
    assert "dinosaur-v3-train" not in json.dumps(meta)
    assert "goodpjw2008/rsna-knee-2-5d-convnext-reader" in meta["dataset_sources"]
    spec_json = json.loads((built / "overlay_spec.json").read_text(encoding="utf-8"))
    assert spec_json["test_submit"] is False
    assert spec_json["do_not_submit_holdout_ids"] is True
    assert spec_json["weight"] == 0.15
    assert spec_json["findings"] == [
        "ACL", "MCL", "Medial Meniscus", "Medial OA", "Effusion", "Synovitis", "Baker's",
    ]


def test_graft_head_package_keeps_gpu_off_and_embeds_the_shipped_export():
    import base64
    import importlib.util
    import json
    import zlib

    builder_path = ROOT / "scripts" / "build_graft_head_kernel.py"
    spec = importlib.util.spec_from_file_location("build_graft_head_kernel", builder_path)
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(builder)
    built = builder.build_kernel()
    embed = builder.EMBED
    meta = json.loads((built / "kernel-metadata.json").read_text(encoding="utf-8"))
    overlay = json.loads((built / "overlay_spec.json").read_text(encoding="utf-8"))
    text = (built / "main.py").read_text(encoding="utf-8")
    assert meta["enable_gpu"] is False
    assert meta["enable_internet"] is False
    assert meta["id"] == "vitaliecervinschi/rsna-knee-graft-head-labels"
    assert "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461" in meta["docker_image"]
    assert "2757e0c7" not in meta["docker_image"]
    assert "pilkwang/rsna-knee-llm-labels" in meta["dataset_sources"]
    assert overlay["test_submit"] is False
    assert overlay["push"] is False
    assert overlay["do_not_submit_holdout_ids"] is True
    assert overlay["keep"] == "55779066"
    assert "DO NOT SUBMIT" in text
    assert "Report" not in text
    ns: dict = {"__name__": "graft_blob"}
    exec(compile(text, str(built / "main.py"), "exec"), ns)
    blobs = ns["BLOBS"]
    for name, path in embed.items():
        raw = zlib.decompress(base64.b64decode(blobs[name]))
        assert raw == path.read_bytes()


def test_champion_blend_metadata_mounts_v5_on_the_anchor_image():
    """Local mill metadata must match the mount that scored 0.936 as 56921398.

    The rejected dinosaur-v3-train slug created an empty-source version 3.
    This test does not push the kernel.
    """
    import json

    meta = json.loads((ROOT / "kaggle_kernel_blend_script" / "kernel-metadata.json").read_text(encoding="utf-8"))
    text = (ROOT / "kaggle_kernel_blend_script" / "main.py").read_text(encoding="utf-8")
    assert meta["id"] == "vitaliecervinschi/rsna-knee-dino-coatnet-script-v1"
    assert meta["kernel_sources"] == ["romantamrazov/rsna-knee-dinosaur-v5-train"]
    assert "dinosaur-v3-train" not in json.dumps(meta)
    assert "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461" in meta["docker_image"]
    assert "2757e0c7" not in meta["docker_image"]
    assert meta["enable_internet"] is False
    assert "walk('test_series')" in text
    assert "Report" not in text


def test_slice_order_follows_the_normal_and_laterality_uses_lps_x():
    from rsna_knee.geometry import laterality_from_lps_x, slice_normal, sort_along_normal

    axial = slice_normal(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    np.testing.assert_allclose(axial, [0.0, 0.0, 1.0])
    flipped = slice_normal(np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(flipped, [0.0, 0.0, -1.0])
    positions = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    order = sort_along_normal(positions, axial)
    assert list(order) == [1, 3, 0, 2]
    assert list(sort_along_normal(positions, flipped)) == [0, 2, 3, 1]
    assert laterality_from_lps_x(12.0) == "L"
    assert laterality_from_lps_x(-3.0) == "R"
    assert laterality_from_lps_x(0.0) == "U"
    try:
        slice_normal(np.zeros(6))
    except ValueError as exc:
        assert "degenerate" in str(exc)
    else:
        raise AssertionError("a zero orientation must fail")


def test_convnext_wheel_install_uses_absolute_paths_and_the_child_can_import_them(tmp_path: Path):
    import os
    import sys

    from rsna_knee.convnext_overlay_kernel import (
        compatible_wheels,
        convnext_child_pythonpath,
        interpreter_wheel_tag,
        wheel_install_argv,
    )

    assets = tmp_path / "assets"
    assets.mkdir()
    names = [
        "pylibjpeg-2.1.0-py3-none-any.whl",
        "pylibjpeg_libjpeg-2.4.0-cp311-cp311-manylinux.whl",
        "pylibjpeg_libjpeg-2.4.0-cp312-cp312-manylinux.whl",
        "pylibjpeg_openjpeg-2.6.0-cp312-cp312-manylinux.whl",
    ]
    wheels = []
    for name in names:
        path = assets / name
        path.write_bytes(b"wheel")
        wheels.append(path)
    chosen = compatible_wheels(wheels, "cp312")
    chosen_names = {path.name for path in chosen}
    assert "pylibjpeg-2.1.0-py3-none-any.whl" in chosen_names
    assert "pylibjpeg_libjpeg-2.4.0-cp312-cp312-manylinux.whl" in chosen_names
    assert "pylibjpeg_openjpeg-2.6.0-cp312-cp312-manylinux.whl" in chosen_names
    assert all("cp311" not in name for name in chosen_names)
    argv = wheel_install_argv(sys.executable, chosen, tmp_path / "_own_env")
    assert "--find-links" not in argv
    assert [item for item in argv if item.endswith(".whl")] == [str(path.resolve()) for path in chosen]
    child = convnext_child_pythonpath(tmp_path / "own_src", tmp_path / "_own_env", "parent")
    assert child.split(os.pathsep)[:2] == [str(tmp_path / "own_src"), str(tmp_path / "_own_env")]
    assert child.endswith("parent")
    assert interpreter_wheel_tag(3, 12) == "cp312"


def test_unequal_k7_overlay_package_is_local_and_fail_closed():
    import base64
    import importlib.util
    import json
    import zlib

    from rsna_knee.convnext_k7_unequal_overlay_kernel import HEADER
    from rsna_knee.fusion import convnext_oof_unequal_weights
    from rsna_knee.schema import SUBMISSION_COLUMNS

    assert HEADER == ",".join(SUBMISSION_COLUMNS)
    assert "Baker's" in HEADER
    weights = convnext_oof_unequal_weights()
    assert weights["Lateral Meniscus"] == 0.0
    assert weights["PF OA"] == 0.0
    assert max(weights.values()) < 0.50
    builder_path = ROOT / "scripts" / "build_convnext_k7_unequal_overlay_kernel.py"
    spec = importlib.util.spec_from_file_location("build_k7_unequal", builder_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    built = module.build_kernel(dst=ROOT / "kaggle_kernel_convnext_k7_unequal_overlay")
    text = (built / "main.py").read_text(encoding="utf-8")
    assert "convnext_k7_unequal_overlay_kernel" in text
    assert "55779066" in text
    public = (ROOT / "data" / "test.csv").read_text(encoding="utf-8")
    for line in public.splitlines()[1:]:
        uid = line.split(",")[0].strip()
        if uid:
            assert uid not in text
    ns: dict = {"__name__": "unequal_blob"}
    exec(compile(text, str(built / "main.py"), "exec"), ns)
    blobs = ns["BLOBS"]
    entry = zlib.decompress(base64.b64decode(blobs["convnext_k7_unequal_overlay_kernel.py"])).decode("utf-8")
    runner = zlib.decompress(base64.b64decode(blobs["convnext_overlay_kernel.py"])).decode("utf-8")
    mill = zlib.decompress(base64.b64decode(blobs["_mill_v10.py"])).decode("utf-8")
    assert "convnext_oof_unequal_weights" in entry
    assert "run_overlay_map" in entry
    assert "Report" not in entry
    assert "fail_closed_rank_blend" in runner
    assert "refuse_over_clock" in runner
    assert "walk('test_series')" in mill
    meta = json.loads((built / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["enable_internet"] is False
    assert "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461" in meta["docker_image"]
    assert meta["kernel_sources"] == ["romantamrazov/rsna-knee-dinosaur-v5-train"]
    assert "dinosaur-v3-train" not in json.dumps(meta)
    spec_json = json.loads((built / "overlay_spec.json").read_text(encoding="utf-8"))
    assert spec_json["test_submit"] is False
    assert spec_json["weights"]["Lateral Meniscus"] == 0.0
    assert spec_json["weights"]["PF OA"] == 0.0
    assert spec_json["weights"]["Effusion"] == 0.49
    assert spec_json["dropped"] == [
        "Lateral Meniscus",
        "Lateral OA",
        "PF OA",
        "Contusion",
        "Fracture",
    ]
