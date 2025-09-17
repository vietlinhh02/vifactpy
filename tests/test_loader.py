from pathlib import Path

from vifact.data.loader import iter_vifact_json


def test_iter_vifact_json_demo():
    data_path = Path("data/raw/ise-dsc01-warmup.json")
    count = 0
    for ex_id, ex in iter_vifact_json(data_path, large=False):
        assert ex_id
        assert isinstance(ex, dict)
        assert "claim" in ex
        count += 1
        if count > 5:
            break
    assert count > 0

