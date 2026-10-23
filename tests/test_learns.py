"""The no-OAI trainer has to move. Fold-0 v2 did not.

16 synthetic studies, 150 optimizer steps, real CoAtNet. Train DICOMs are not
on this machine, so the batches are tensors in the train window shape at 160 px
(the kernel trains at 320). Pass: loss drops at least 30% and probability
standard deviation is above 0.05.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from rsna_knee.raptor_model import ARCH, RaptorClassifier, imagenet_norm
from rsna_knee.raptor_train import ACCUM, Stepper, make_optimizer, probability_std, weighted_bce

STEPS = 150
N_STUDIES = 16


def _studies(device):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(2026)
    batches = []
    for i in range(N_STUDIES):
        label = float(i % 2)
        shift = 1.5 if label > 0.5 else -1.5
        windows = torch.randn(1, 4, 3, 160, 160, generator=gen) + shift
        batches.append((
            imagenet_norm(windows).to(device),
            torch.full((1, 12), label, device=device),
            torch.ones(1, 12, device=device),
        ))
    return batches


def test_coatnet_loss_drops_and_predictions_spread():
    if not torch.cuda.is_available():
        raise RuntimeError("RTX learn test needs CUDA")
    import timm
    torch.backends.cudnn.benchmark = True
    backbone = timm.create_model(
        ARCH, pretrained=True, num_classes=0, in_chans=3, global_pool="avg", img_size=160,
    )
    model = RaptorClassifier(backbone, f_dim=backbone.num_features, n=12).cuda()
    stepper = Stepper(make_optimizer(model), torch.amp.GradScaler("cuda"), accum=ACCUM)
    batches = _studies("cuda")
    model.train()
    stepper.opt.zero_grad(set_to_none=True)
    losses = []
    probs = []
    i = 0
    while stepper.updates < STEPS:
        x, y, w = batches[i % N_STUDIES]
        i += 1
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = weighted_bce(logits, y, w)
        stepper.backward(loss)
        if stepper.ready():
            losses.append(float(loss.detach()))
            stepper.update(warmup=10, total=STEPS)
        if stepper.updates == STEPS:
            break
    model.eval()
    with torch.no_grad():
        for x, _y, _w in batches:
            probs.append(torch.sigmoid(model(x.float())).float().cpu())
    first, last = losses[0], losses[-1]
    std = probability_std(torch.cat(probs, 0))
    assert last <= first * 0.70, f"loss {first:.4f} -> {last:.4f}"
    assert std > 0.05, f"pred std {std:.5f}"
