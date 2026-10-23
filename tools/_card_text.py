import json
from pathlib import Path

root = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\artifacts\provenance\card_scan")
want = [
    "dreaddevelopment__raptor-knee-finespacing",
    "tannguyenngoc__rsna-knee-resnet34-attention-v3-checkpoint",
    "eduardozw__rsna-knee-weights",
    "heliosli__rsna-knee-compact-v1",
    "heliosli__rsna-knee-compact-r7cnxfsss",
    "tannguyenngoc__rsna-knee-coatnet-a100-best",
    "octaviograu__rsna-knee-wide336-frozen-heads-v1",
    "prvsiyan__rsna-knee-b3-v47-public-deployment",
]
for name in want:
    meta = json.loads((root / name / "dataset-metadata.json").read_text(encoding="utf-8"))
    info = meta.get("info", meta)
    desc = (info.get("description") or "").replace("\n", " ")
    print("=" * 20, name)
    print(desc[:1200])
