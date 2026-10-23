# Clean weight candidates

Searched public RSNA Knee datasets on 2026-10-14. A card is clean only when it states the training labels, those labels are not OAI, and gold58 is not in the training loss. Cards that skip the label sentence are excluded. Holdout870 is the seed-2026 870-study split. No new inference was run.

## Top

| # | Slug | License | Training labels on the card | Local holdout870 |
|---|---|---|---|---|
| 1 | `goodpjw2008/rsna-knee-2-5d-convnext-reader` | Apache-2.0 | Report-derived soft labels. The 58 gold studies were not used for training. | 3-fold reader **0.9439** macro. Public fold0 submit **0.922**. Author reports **0.929** for the 3-fold reader. |
| 2 | `dreaddevelopment/raptor-knee-widedense` | CC0-1.0 | LLM report soft labels on 4,349 competition studies. Gold58 held out. Card text: "No external data was used." Claims 0.924 public and 0.9167 on the 58 gold studies. | No local prediction file. Not scored. |
| 3 | none | | No third public weight card states both the label source and that gold58 stayed out of the loss. | |

## Excluded on this pass

- `dreaddevelopment/raptor-knee-maxspan` and `dreaddevelopment/raptor-knee-finespacing`: geometry and public scores only. The cards do not state the training labels.
- `heliosli/rsna-knee-compact-v1`: says "report-derived targets" and does not say whether gold58 was held out.
- `nartaa/rsna-knee-publication-swa-weights-20261007`: card names OAI.
- `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`, `antoinegg1` E9/E11, `romantamrazov/rsna-knee-dinosaur-v5-train`, `prvsiyan/rsna-knee-b3-v47-public-deployment`: gold58-fit or gold58-audit heads, or the card does not clear them.
- `riadmohamed42/jev-knee-labels`: JEV HYBRID. Not a weight card for this pool.
