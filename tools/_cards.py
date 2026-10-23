import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

SLUGS = [
    "dreaddevelopment/raptor-knee-widedense",
    "dreaddevelopment/raptor-knee-finespacing",
    "dreaddevelopment/raptor-knee-maxspan",
    "crtproduction/rsna-knee-effb0-mil-weights",
    "kitopl/rsna-knee-eda-baseline-weights",
    "tannguyenngoc/rsna-knee-resnet34-attention-v3-checkpoint",
    "eduardozw/rsna-knee-weights",
    "easonyanyan/rsna-knee-c12-weights",
    "heliosli/rsna-knee-compact-v1",
    "heliosli/rsna-knee-compact-r7cnxfsss",
    "myominhtet/rsna-knee-coat5fold",
    "lixin73/rsna-knee-train-checkpoints",
    "lixin73/rsna-knee-mavit-336px-v51-weights",
    "mattiaangeli/rsna-knee-cnx-m448-f0-public",
    "tannguyenngoc/rsna-knee-coatnet-a100-best",
    "haideptry/rsna-knee-coatnet-25d-masked-checkpoint",
    "starkhushi/rsna-knee-v6f1-weights",
    "zaidaliiq1000/rsna-knee-ft3-weights",
    "octaviograu/rsna-knee-wide336-frozen-heads-v1",
    "prvsiyan/rsna-knee-b3-v47-public-deployment",
    "nartaa/rsna-knee-publication-swa-weights-20261007",
    "goodpjw2008/rsna-knee-2-5d-convnext-reader",
]
api = KaggleApi()
api.authenticate()
root = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\artifacts\provenance\card_scan")
root.mkdir(parents=True, exist_ok=True)
for slug in SLUGS:
    dest = root / slug.replace("/", "__")
    try:
        api.dataset_metadata(slug, str(dest))
        meta = json.loads((dest / "dataset-metadata.json").read_text(encoding="utf-8"))
        info = meta.get("info", meta)
        desc = info.get("description") or ""
        lic = info.get("licenses")
        low = desc.lower()
        flags = []
        for token in ("oai", "external", "gold", "4,349", "4349", "report", "llm", "imagenet", "label"):
            if token in low:
                flags.append(token)
        print("OK", slug, "lic", lic, "flags", ",".join(flags), "desc_len", len(desc))
    except Exception as exc:
        print("ERR", slug, type(exc).__name__, str(exc)[:160])
