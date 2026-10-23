"""Drive shipped gold-AUC ≫ 0.94 leak/overfit abort."""

import pytest

from rsna_knee.v4_split import (
    GOLD_AUC_ABORT_THRESHOLD,
    GoldOverfitAbort,
    abort_if_gold_overfit,
    should_abort_gold_overfit,
)


def test_threshold_is_point_nine_four():
    assert GOLD_AUC_ABORT_THRESHOLD == 0.94


def test_macro_clearly_above_threshold_aborts():
    assert should_abort_gold_overfit(0.99) is True
    assert should_abort_gold_overfit(0.9575) is True
    with pytest.raises(GoldOverfitAbort, match="0.9900"):
        abort_if_gold_overfit(0.99)


def test_macro_clearly_below_threshold_does_not_abort():
    assert should_abort_gold_overfit(0.85) is False
    assert should_abort_gold_overfit(0.94) is False
    abort_if_gold_overfit(0.85)
    abort_if_gold_overfit(0.94)
