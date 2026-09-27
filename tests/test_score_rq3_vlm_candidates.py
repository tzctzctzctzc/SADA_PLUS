import subprocess
import sys
import types

import pytest

from tools import score_rq3_vlm_candidates as score


def test_interface_preflight_checks_exact_clean_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(score, "CODE_ROOT", tmp_path)
    monkeypatch.setattr(
        score.subprocess, "check_output",
        lambda command, cwd, text: "abc123\n" if command[1:3] == ["rev-parse", "HEAD"] else "",
    )
    score.check_code("abc123")

    monkeypatch.setattr(
        score.subprocess, "check_output",
        lambda command, cwd, text: "abc123\n" if command[1:3] == ["rev-parse", "HEAD"] else " M dirty.py\n",
    )
    with pytest.raises(ValueError, match="clean code commit"):
        score.check_code("abc123")


def test_interface_preflight_rejects_busy_gpu_without_interruption(monkeypatch):
    def query(command, text):
        if any(argument.startswith("--query-gpu=") for argument in command):
            return "GPU-3, 2048, 0\n"
        return ""

    monkeypatch.setattr(score.subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="external jobs are not interrupted"):
        score.idle_gpu_uuid(3)


def test_ego_requires_exact_clean_openai_clip_revision(monkeypatch, tmp_path):
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "__init__.py").write_text("")

    def query(command, cwd, text):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return score.OPENAI_CLIP_REVISION + "\n"
        return ""

    monkeypatch.setattr(score.subprocess, "check_output", query)
    assert score.openai_clip_root({"clip_root": str(tmp_path)}) == tmp_path

    monkeypatch.setattr(
        score.subprocess, "check_output",
        lambda command, cwd, text: "unknown\n" if command[1:3] == ["rev-parse", "HEAD"] else "",
    )
    with pytest.raises(ValueError, match="pinned clean OpenAI CLIP"):
        score.openai_clip_root({"clip_root": str(tmp_path)})


def test_intern_import_preserves_parent_for_official_relative_utils(monkeypatch, tmp_path):
    calls = []

    def load(name):
        calls.append(name)
        module = types.SimpleNamespace()
        if name.endswith('internvideo2_clip_vision'):
            module.InternVideo2 = object()
        if name.endswith('internvideo2_clip_text'):
            module.LLaMA, module.Tokenizer = object(), object()
        return module

    monkeypatch.setattr(score.importlib, 'import_module', load)
    # namespace初始化本身不执行官方__init__；只有这四个确切模块被请求。
    existing = set(sys.modules)
    try:
        score.import_internvideo_teacher(tmp_path)
        assert calls == ['flash_attn.ops.rms_norm',
                         '_rq3_official_intern.models.backbones.internvideo2.internvideo2_clip_vision',
                         '_rq3_official_intern.models.backbones.internvideo2.internvideo2_clip_text',
                         '_rq3_official_intern.models.internvideo2_clip']
        assert sys.modules['_rq3_official_intern.utils'].__path__ == [str(tmp_path / 'utils')]
    finally:
        for name in set(sys.modules) - existing:
            if name.startswith('_rq3_official_intern'):
                del sys.modules[name]
