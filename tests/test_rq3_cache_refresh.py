from pathlib import Path

from tools.run_rq3_trusted_queue import prepare_cache_overlay


def test_refresh_overlay_never_reuses_scheduled_teacher(tmp_path):
    original = tmp_path / 'old'
    for model in ('qwen3_vl_8b', 'internvideo2_clip_1b'):
        (original / model).mkdir(parents=True)
        for shift in ('S1', 'S2'):
            (original / model / (shift + '.json')).write_text(model + shift)
    rows = [dict(teacher='qwen3_vl_8b', shift='S2', strategy=s)
            for s in ('verify', 'prototype')]
    root = prepare_cache_overlay(tmp_path / 'new', original, rows)
    assert not (root / 'qwen3_vl_8b/S2.json').exists()
    assert (root / 'qwen3_vl_8b/S1.json').is_symlink()
    assert (root / 'internvideo2_clip_1b/S2.json').read_text() == 'internvideo2_clip_1bS2'
    assert (original / 'qwen3_vl_8b/S2.json').read_text() == 'qwen3_vl_8bS2'


def test_formal_runner_and_prototype_share_refreshed_root():
    import inspect
    from tools.run_rq3_trusted_queue import main
    source = inspect.getsource(main)
    assert "cache = cache_root / model" in source
    assert "'--cache-root', str(cache_root)" in source
    assert 'args.prepared_root and not args.refresh_teacher_caches' in source
