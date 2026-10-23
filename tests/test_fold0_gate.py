"""Fold-0 kill requires mill on the same studies. A partial holdout does not kill."""

import numpy as np
import pandas as pd

from rsna_knee.fold0_gate import fold0_decision
from rsna_knee.schema import FINDINGS, ID_COLUMN


def _frame(ids, scores):
    rows = []
    for sid, row in zip(ids, scores):
        item = {ID_COLUMN: sid}
        item.update({name: float(v) for name, v in zip(FINDINGS, row)})
        rows.append(item)
    return pd.DataFrame(rows)


def test_partial_mill_does_not_kill_and_full_coverage_can():
    ids = ["a", "b", "c", "d"]
    labels = np.tile(np.array([0.0, 1.0]), 6)
    # Two positives and two negatives on every finding, alternating by study.
    y = np.vstack([labels, 1 - labels, labels, 1 - labels])
    weak = 1.0 - y
    strong = y.copy()
    fold = set(ids)
    partial = fold0_decision(_frame(ids, strong), _frame(ids, y), _frame(ids[:2], strong), fold)
    assert partial["decision"] == "incomplete"
    assert partial["mill_n"] == 2
    kill = fold0_decision(_frame(ids, weak), _frame(ids, y), _frame(ids, strong), fold)
    assert kill["decision"] == "kill"
    assert kill["oof_macro"] < kill["mill_macro"]
