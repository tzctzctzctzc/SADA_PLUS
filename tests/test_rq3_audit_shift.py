import json
from pathlib import Path

import pytest

from tools.audit_rq3_revised_cpu import audit_shift_inputs


@pytest.mark.parametrize('shift,folder', [
    ('S1', 'dark_vs_all_new_videos_full'), ('S2', 'dark_vs_all_old_videos_full'),
    ('S3', 'white_vs_all_new_videos_full'), ('S4', 'white_vs_all_old_videos_full'),
    ('S5', 'old_vs_new'), ('S6', 'new_vs_old'),
])
def test_cpu_audit_uses_the_cache_shift(tmp_path, shift, folder):
    cache = tmp_path / 'cache.json'
    cache.write_text(json.dumps({'shift': shift}))
    actual, config = audit_shift_inputs(cache, Path(__file__).resolve().parents[1])
    assert actual == shift
    assert config.parent.name == folder
    assert config.is_file()


def test_cpu_audit_rejects_unknown_shift(tmp_path):
    cache = tmp_path / 'cache.json'
    cache.write_text('{"shift": "S7"}')
    with pytest.raises(ValueError, match='unknown cache shift'):
        audit_shift_inputs(cache, Path(__file__).resolve().parents[1])
