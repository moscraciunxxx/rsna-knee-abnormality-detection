#!/usr/bin/env python
"""Inspect a nursrijan-style DINOv2-B LLM-finetune .pt for a 12-head overlay.

Mill (`kaggle_kernel_blend_script/main.py`) boots DINO with
`_boot_find_dir(['config.json'])`. A dataset that drops `config.json` at its
root will steal that pointer from `metaresearch/dinov2/pytorch/small/1`.
dino3s already rewired the marker to `pytorch_model.bin` / `model.safetensors`.

This script does not download, does not push a kernel, and does not invent a
head. Pass a local .pt (e.g. `.cache/dinov2b/r336_best.pt`) or run with no
path to print the keys a loadable 12-head checkpoint must contain.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
FINDINGS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
EXPECTED_TOP = ("model", "cfg", "auc", "annot_auc")
EXPECTED_HEAD = (
    "head.slot_emb",
    "head.query",
    "head.proj.0.weight",
    "head.proj.0.bias",
    "head.proj.1.weight",
    "head.proj.1.bias",
    "head.out.weight",
    "head.out.bias",
)
THEFT_MARKERS = ("config.json",)
MILL_BOOT_MARKERS = ("pytorch_model.bin", "model.safetensors")
FORBIDDEN_ROOT_NAMES = ("config.json",)


def _print(title: str, body: str = "") -> None:
    print(f"== {title} ==")
    if body:
        print(body.rstrip())
        print()


def expected_keys_report() -> None:
    _print(
        "expected checkpoint layout (no .pt given)",
        "\n".join(
            [
                "top-level: dict with keys model / cfg / auc / annot_auc",
                "cfg: {'name': 'r336'|'r224', 'img': 336|224}",
                "model: state_dict, mill Model + HF DINOv2-base SlotHead",
                "  mean, std                          ImageNet buffers (1,3,1,1)",
                "  backbone.embeddings.cls_token      (1,1,768)",
                "  backbone.embeddings.position_embeddings  (1,1370,768) native 518 pos",
                "  backbone.embeddings.patch_embeddings.projection.weight  (768,3,14,14)",
                "  backbone.encoder.layer.{0..11}.*   12 ViT-B blocks, hidden 768",
                "  backbone.layernorm.{weight,bias}   (768,)",
                "  head.slot_emb                      (6,256)   6 recovered slots",
                "  head.query                         (12,256)  12 findings",
                "  head.proj.0.{weight,bias}          LayerNorm(1536) = 768*cls_mean",
                "  head.proj.1.{weight,bias}          Linear(1536,256)",
                "  head.out.{weight,bias}             (12,256) / (12,)",
                "absent: head.slot_prior  (train used prior=False)",
                "absent: config.json next to the .pt (mill boot theft)",
            ]
        ),
    )
    _print(
        "mill discovery (kaggle_kernel_blend_script/main.py)",
        "\n".join(
            [
                "DINO = _boot_find_dir(['config.json'], extra=(small HF paths,))",
                "find_dinov2(variant) ignores variant and returns that DINO dir",
                "AutoModel.from_pretrained(DINO) is the mill 20-member backbone",
                "first BFS hit of config.json under /kaggle/input wins",
                "alphabetical dinov2/pytorch/base beats dinov2/pytorch/small",
                "dino3s patch: markers pytorch_model.bin + model.safetensors",
                "new overlay dataset MUST NOT put config.json at dataset root",
                "do NOT attach metaresearch/dinov2/PyTorch/base/1 beside mill small",
                "require mill boot to keep pointing at small/*.bin or *.safetensors",
            ]
        ),
    )


def _as_state(blob: object) -> tuple[dict, dict]:
    meta: dict = {"top_type": type(blob).__name__}
    if not isinstance(blob, dict):
        raise TypeError(f"checkpoint is {type(blob).__name__}, expected dict")
    meta["top_keys"] = list(blob.keys())
    for key in ("cfg", "auc", "annot_auc", "config", "args"):
        if key in blob and not hasattr(blob[key], "shape"):
            meta[key] = blob[key]
    sd = None
    for cand in ("model", "model_state", "state_dict", "net", "ema", "weights"):
        if cand in blob and isinstance(blob[cand], dict):
            tensor_keys = [k for k, v in blob[cand].items() if hasattr(v, "shape")]
            if tensor_keys:
                sd = blob[cand]
                meta["state_container"] = cand
                break
    if sd is None:
        tensor_keys = [k for k, v in blob.items() if hasattr(v, "shape")]
        if tensor_keys:
            sd = {k: blob[k] for k in tensor_keys}
            meta["state_container"] = "<top-level tensors>"
    if sd is None:
        raise TypeError("no tensor state_dict in checkpoint")
    return sd, meta


def _shape(t) -> tuple:
    return tuple(t.shape) if hasattr(t, "shape") else ()


def analyze_state(sd: dict, meta: dict) -> dict:
    keys = list(sd.keys())
    prefixes = collections.Counter(k.split(".")[0] for k in keys)
    layer_ids = []
    for k in keys:
        parts = k.split(".")
        if len(parts) > 4 and parts[0] == "backbone" and parts[1] == "encoder" and parts[2] == "layer":
            try:
                layer_ids.append(int(parts[3]))
            except ValueError:
                pass
    n_layer = (max(layer_ids) + 1) if layer_ids else 0
    hidden = None
    patch = None
    pos_tokens = None
    cls = sd.get("backbone.embeddings.cls_token")
    if cls is not None and cls.ndim >= 1:
        hidden = int(cls.shape[-1])
    proj = sd.get("backbone.embeddings.patch_embeddings.projection.weight")
    if proj is not None and proj.ndim == 4:
        hidden = int(proj.shape[0])
        patch = int(proj.shape[-1])
        if proj.shape[1] != 3:
            meta["in_chans"] = int(proj.shape[1])
    pos = sd.get("backbone.embeddings.position_embeddings")
    if pos is not None and pos.ndim >= 2:
        pos_tokens = int(pos.shape[1])
        hidden = int(pos.shape[-1])
    query = sd.get("head.query")
    out_w = sd.get("head.out.weight")
    slot_emb = sd.get("head.slot_emb")
    n_findings = int(query.shape[0]) if query is not None else (
        int(out_w.shape[0]) if out_w is not None else 0
    )
    n_slots = int(slot_emb.shape[0]) if slot_emb is not None else 0
    head_hidden = int(query.shape[1]) if query is not None else 0
    proj0 = sd.get("head.proj.0.weight")
    pool_dim = int(proj0.shape[0]) if proj0 is not None else 0
    pool_parts = None
    if hidden and pool_dim:
        if pool_dim % hidden == 0:
            pool_parts = pool_dim // hidden
    has_prior = "head.slot_prior" in sd
    has_head = any(k.startswith("head.") for k in keys)
    encoder_only = (not has_head) and any(k.startswith("backbone.") or "patch_embed" in k for k in keys)
    img = None
    cfg = meta.get("cfg") if isinstance(meta.get("cfg"), dict) else {}
    if "img" in cfg:
        img = int(cfg["img"])
    elif pos_tokens and patch:
        patches = pos_tokens - 1
        side = int(round(patches ** 0.5))
        if side * side == patches:
            img = side * patch
            meta["pos_embed_native_res"] = img
    vit = None
    if hidden == 768 and patch == 14 and n_layer == 12:
        vit = "ViT-B/14 (DINOv2-base)"
    elif hidden == 384 and patch == 14 and n_layer == 12:
        vit = "ViT-S/14 (DINOv2-small)"
    elif hidden == 384 and patch == 16:
        vit = "ViT-S/16 (not DINOv2-B)"
    twelve = bool(has_head and n_findings == 12 and not encoder_only)
    return {
        "n_keys": len(keys),
        "first_40_keys": keys[:40],
        "prefixes": prefixes.most_common(),
        "architecture": vit or f"unknown hidden={hidden} patch={patch} layers={n_layer}",
        "hidden": hidden,
        "patch": patch,
        "n_layer": n_layer,
        "pos_tokens": pos_tokens,
        "n_findings": n_findings,
        "n_slots": n_slots,
        "head_hidden": head_hidden,
        "pool_dim": pool_dim,
        "pool_parts": pool_parts,
        "pool_name": {2: "cls_mean", 3: "cls_mean_focal"}.get(pool_parts, "unknown"),
        "has_slot_prior": has_prior,
        "twelve_head": twelve,
        "encoder_only": encoder_only,
        "img": img,
        "cfg_name": cfg.get("name"),
        "auc": meta.get("auc"),
        "annot_auc": meta.get("annot_auc"),
        "missing_expected_head": [k for k in EXPECTED_HEAD if k not in sd],
        "n_params_backbone": int(sum(v.numel() for k, v in sd.items() if k.startswith("backbone") and hasattr(v, "numel"))),
        "n_params_head": int(sum(v.numel() for k, v in sd.items() if k.startswith("head") and hasattr(v, "numel"))),
        "has_registers": any("register" in k for k in keys),
    }


def sibling_theft(pt_path: Path) -> dict:
    folder = pt_path.parent
    names = sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []
    root_config = (folder / "config.json").is_file()
    mill_weight = any((folder / n).is_file() for n in MILL_BOOT_MARKERS)
    return {
        "folder": str(folder),
        "names": names,
        "root_config_json": root_config,
        "has_pytorch_model_bin_or_safetensors": mill_weight,
        "theft_risk": "HIGH" if root_config else "LOW (no config.json beside .pt)",
        "mill_boot_safe_to_attach": (not root_config) and (not mill_weight),
    }


def load_blob(path: Path):
    import torch

    last = None
    for weights_only in (True, False):
        try:
            blob = torch.load(path, map_location="cpu", weights_only=weights_only)
            return blob, {"map_location": "cpu", "weights_only": weights_only}
        except Exception as exc:
            last = exc
    raise RuntimeError(f"torch.load failed: {last}") from last


def report(path: Path | None) -> int:
    expected_keys_report()
    if path is None:
        print("pass a local .pt to inspect keys; nothing was loaded")
        return 0
    path = path.expanduser().resolve()
    if not path.is_file():
        print(f"missing file: {path}", file=sys.stderr)
        return 2
    print(f"file: {path}")
    print(f"size_bytes: {path.stat().st_size}")
    theft = sibling_theft(path)
    _print("config.json theft", json.dumps(theft, indent=2))
    blob, how = load_blob(path)
    print(f"loaded_with: {how}")
    sd, meta = _as_state(blob)
    info = analyze_state(sd, meta)
    _print(
        "architecture",
        "\n".join(
            [
                f"arch: {info['architecture']}",
                f"hidden: {info['hidden']}",
                f"patch: {info['patch']}",
                f"n_layer: {info['n_layer']}",
                f"pos_tokens: {info['pos_tokens']}",
                f"registers: {info['has_registers']}",
                f"backbone_params: {info['n_params_backbone']}",
                f"head_params: {info['n_params_head']}",
            ]
        ),
    )
    _print(
        "heads / output",
        "\n".join(
            [
                f"twelve_head: {info['twelve_head']}",
                f"encoder_only: {info['encoder_only']}",
                f"n_findings: {info['n_findings']}  (competition has {len(FINDINGS)})",
                f"n_slots: {info['n_slots']}",
                f"head_hidden: {info['head_hidden']}",
                f"pool: {info['pool_name']} dim={info['pool_dim']} parts={info['pool_parts']}",
                f"slot_prior in state: {info['has_slot_prior']}  (load with prior=False if False)",
                f"missing_expected_head: {info['missing_expected_head']}",
            ]
        ),
    )
    _print(
        "resolution / scores",
        "\n".join(
            [
                f"cfg.name: {info['cfg_name']}",
                f"cfg.img / inferred: {info['img']}",
                f"auc: {info['auc']}",
                f"annot_auc: {info['annot_auc']}",
            ]
        ),
    )
    print("== first 40 keys ==")
    for i, k in enumerate(info["first_40_keys"], 1):
        shp = _shape(sd[k])
        print(f"{i:02d}  {k}  {shp}")
    print()
    print("== loadable 12-head? ==")
    ok = (
        info["twelve_head"]
        and info["hidden"] == 768
        and info["patch"] == 14
        and info["n_findings"] == 12
        and not info["encoder_only"]
        and not info["missing_expected_head"]
    )
    print("yes" if ok else "no")
    if ok:
        print(
            "instantiate HF DINOv2-base from in-code Dinov2Config "
            "(do not AutoModel.from_pretrained a directory mill can BFS), "
            "SlotHead(dim=1536, n_slot=6, n_out=12, prior=False), "
            "load_state_dict(ck['model'], strict=True), img=cfg['img']."
        )
    else:
        print("missing a 12-finding SlotHead and/or ViT-B/14 backbone; do not fake a head")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="local .pt (r336_best.pt). omit to print required keys only",
    )
    args = parser.parse_args(argv)
    path = Path(args.checkpoint) if args.checkpoint else None
    return report(path)


if __name__ == "__main__":
    raise SystemExit(main())
