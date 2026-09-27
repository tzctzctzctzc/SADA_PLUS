import subprocess

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
