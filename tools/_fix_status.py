from pathlib import Path

root = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection")
block = """## coatnet MCL+Effusion TEST overlay PUSHED @ 2026-10-07 ~08:25 PT

- **Slug:** `vitaliecervinschi/rsna-knee-coatnet-mcl-effusion-ov` (version 3)
- **Local folder:** `D:\\Coding Compete\\RSNA Knee Abnormality Detection\\kaggle_kernel_coatnet_mcl_eff_overlay`
- **Job:** mill V10 base + CoAtNet stack, licensed overlay **MCL+Effusion @0.15** (`artifacts/coatnet870/overlay_spec.json`)
- **Guards:** hard watchdog `os._exit(99)` at 8.75h; `kaggle kernels push -t 32000`; coatnet timeout fail-closed to mill
- **Not** `rsna-knee-coatnet-test-v1` (full CoAt replace on 3 public studies, COMPLETE, no mill blend — wrong product)
- **Status at write:** RUNNING (Save Version). **Do NOT Submit** without Vitaly OK. Keep anchor **55779066 / 0.936**.
- **GPU:** ~8.26h used / **21.74h** remaining, refresh 2026-10-10T00:00 UTC
- **Zero Submits** this session.
- v1 ERROR: side `_mill_v10.py` not uploaded by Kaggle (only code_file). v2 ERROR: mill ASSET boot matched mattiaangeli/knee-mri-fold-weights. v3: payloads embedded in main.py + ASSET prefers tonylica `bundle_manifest.json`.

### Next bets queue (independent of Submit OK)

1. **Watch this overlay to COMPLETE** — confirm log line `OVERLAY MCL+Effusion w=0.15`. Then wait Vitaly OK before any Submit. If publicScore <= 0.936 after a future Submit, keep 55779066.
2. **CPU / no-GPU:** inventory unused public heads not in mill or this coat stack; draft a private 870 gate only for a non-ruled-out family (not PF OA+Synovitis @0.20, not native384 V41/logfix, not dino3s 1.0 MCL+Baker, not jointline, not uniform 0.50-all-12).
3. **Do not** re-push coatnet-test-v1 or coatnet870-gate. Do not Submit holdout kernels.
4. **If overlay ERROR again:** pull logs, fix, re-push once; do not burn duplicate GPU on a known-bad kernel.

"""

for name in ("WIN_STATUS.md", "STATUS_OVERNIGHT.md"):
    path = root / name
    prev = path.read_text(encoding="utf-8", errors="replace")
    # Drop previously corrupted inserted section if present
    marker = "## coatnet MCL+Effusion TEST overlay PUSHED"
    if marker in prev:
        # remove from marker through next ## at column 0 after it, or through "### Next bets" block end before old content
        start = prev.find(marker)
        # find next top-level ## after the block (not ###)
        rest = prev[start + len(marker):]
        cut = None
        for i, line in enumerate(rest.splitlines(True)):
            if i == 0:
                continue
            if line.startswith("## ") and not line.startswith("###"):
                # compute absolute
                cut = start + len(marker) + sum(len(x) for x in rest.splitlines(True)[:i])
                break
        if cut is None:
            prev = prev[:start]
        else:
            prev = prev[:start] + prev[cut:]
    if prev.startswith("#"):
        lines = prev.splitlines(True)
        out = []
        inserted = False
        for i, line in enumerate(lines):
            out.append(line)
            if (not inserted) and i > 0 and line.strip() == "":
                out.append(block)
                inserted = True
        if not inserted:
            out = [lines[0], "\n", block] + lines[1:]
        path.write_text("".join(out), encoding="utf-8")
    else:
        path.write_text(block + prev, encoding="utf-8")
    print("fixed", name)

print("WIN_HEAD:")
print((root / "WIN_STATUS.md").read_text(encoding="utf-8")[:1200])
