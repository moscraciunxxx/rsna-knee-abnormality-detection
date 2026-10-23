from pathlib import Path
root = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection")
bt = chr(96)  # backtick
block = f"""## SUBMITTED coatnet MCL+Effusion overlay @ 2026-10-07 ~08:36 PT

- **Kernel:** {bt}vitaliecervinschi/rsna-knee-coatnet-mcl-effusion-ov{bt} v3 COMPLETE
- **OVERLAY log confirmed:** {bt}[mcl-eff-ov] OVERLAY MCL+Effusion w=0.15 on mill V10; shape=(3, 13); elapsed=507s; not 1.0 replace; not r336ov; not Submit without Vitaly OK; keep 55779066.{bt}
- **Submit:** {bt}kaggle competitions submit rsna-knee-abnormality-detection -k vitaliecervinschi/rsna-knee-coatnet-mcl-effusion-ov -v 3 -f submission.csv{bt}
- **Submission id:** **56915024** — status **PENDING** at submit time (hidden-test scoring; publicScore not yet available; n=3 Save Version was expected placeholder)
- **Anchor kept:** **55779066 / 0.936** (not deleted)
- **Did not submit:** coatnet-test-v1, coatnet870-gate, any holdout
- **GPU:** last capture before submit ~**21.59h** left (8.41h used), refresh 2026-10-10T00:00 UTC; hidden scoring will consume more
- **Next:** wait for 56915024 publicScore; if <=0.936 keep 55779066 selected. Queue next: ConvNeXt MIL 870-gate when GPU lane free.

"""
marker = "## SUBMITTED coatnet MCL+Effusion overlay"
for name in ("WIN_STATUS.md", "STATUS_OVERNIGHT.md"):
    path = root / name
    text = path.read_text(encoding="utf-8", errors="replace")
    # strip any prior broken SUBMITTED section (incl VT corruption)
    while marker in text:
        start = text.find(marker)
        rest = text[start + len(marker):]
        cut = None
        offset = 0
        for line in rest.splitlines(True):
            if offset > 0 and line.startswith("## ") and not line.startswith("###"):
                cut = start + len(marker) + offset
                break
            offset += len(line)
        if cut is None:
            text = text[:start]
        else:
            text = text[:start] + text[cut:]
    lines = text.splitlines(True)
    if not lines:
        new = block
    else:
        out = [lines[0]]
        i = 1
        if i < len(lines) and lines[i].strip() == "":
            out.append(lines[i]); i += 1
        out.append(block)
        out.extend(lines[i:])
        new = "".join(out)
    path.write_text(new, encoding="utf-8")
    assert "\x0b" not in new
    print("OK", name, "head:")
    print(new[:650])
    print("---")
