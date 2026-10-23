from pathlib import Path

src = Path(
    r"D:\Coding Compete\RSNA Knee Abnormality Detection\refs"
    r"\dreaddevelopment__knee-mri-twelve-findings-from-a-single-model"
    r"\knee-mri-twelve-findings-from-a-single-model.py"
)
dst_dir = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\kaggle_kernel_clean_blend")
dst_dir.mkdir(exist_ok=True)
text = src.read_text(encoding="utf-8")
old = """ARMS = [
    {"file": "raptor_ft_coatnet_v4_full.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},
]"""
new = """ARMS = [
    {"file": os.environ.get("CLEAN_CKPT", "raptor_ft_coatnet_v4_full.pt"), "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},
]
SPAN_LO = float(os.environ.get("CLEAN_SPAN_LO", "0.06"))
SPAN_HI = float(os.environ.get("CLEAN_SPAN_HI", "0.94"))"""
if old not in text:
    raise SystemExit("arms block missing")
text = text.replace(old, new, 1)
old2 = "n = len(files); lo, hi = int(n * 0.06), int(n * 0.94) - 1; hi = max(hi, lo)"
new2 = "n = len(files); lo, hi = int(n * SPAN_LO), int(n * SPAN_HI) - 1; hi = max(hi, lo)"
if old2 not in text:
    raise SystemExit("span line missing")
text = text.replace(old2, new2, 1)
needle = "    # WEIGHTED rank-mean blend across the test set, per finding (the offline recipe)."
insert = """    if os.environ.get("CLEAN_RAW") == "1":
        raw = pd.DataFrame(np.clip(arm_probs[0], 0, 1).astype(np.float32), columns=LAB)
        raw.insert(0, "StudyInstanceUID", test_ids)
        out = os.environ.get("CLEAN_OUT", "/kaggle/working/arm.csv")
        raw.to_csv(out, index=False)
        print("wrote raw", out, len(raw), flush=True)
        return

"""
if needle not in text:
    raise SystemExit("blend marker missing")
text = text.replace(needle, insert + needle, 1)
(dst_dir / "dread_infer.py").write_text(text, encoding="utf-8", newline="\n")
print("bytes", (dst_dir / "dread_infer.py").stat().st_size)
