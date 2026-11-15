import pandas as pd

from src.utils import load_config
from src.validation.validate import apply_rules, interaction_rules


def _raw(rows):
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"]).astype(str)
    df["_batch_id"], df["_ingested_at"] = "b1", "t"
    return df


def test_interaction_rules_quarantine_bad_records():
    cfg = load_config()
    ok_ts = "880000000"  # 1997-11-20
    df = _raw([
        [1, 10, 4, ok_ts],          # valid
        [1, 10, 4, ok_ts],          # exact duplicate -> warning only
        [2, 11, 9, ok_ts],          # invalid rating
        [2, 11, "x", ok_ts],        # wrong type
        [3, 12, 3, "0"],            # invalid timestamp (1970)
        [999, 10, 3, ok_ts],        # unknown user (referential integrity)
        [1, 777, 3, ok_ts],         # unknown item (referential integrity)
        [1, 12, "", ok_ts],         # missing rating
    ])
    valid, invalid, stats = apply_rules("interactions", df, interaction_rules(df, cfg, {1, 2, 3}, {10, 11, 12}), ["user_id", "item_id"])
    assert len(valid) == 2 and len(invalid) == 6
    reasons = ";".join(invalid["error_reasons"])
    for rule in ["invalid_rating", "invalid_type_rating", "invalid_timestamp", "unknown_user_fk", "unknown_item_fk", "missing_rating"]:
        assert rule in reasons
    dup = next(r for r in stats["rules"] if r["rule"] == "exact_duplicate")
    assert dup["failed_rows"] == 1 and dup["severity"] == "warning"
