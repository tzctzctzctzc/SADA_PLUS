import pickle
from types import SimpleNamespace

import pytest

from tools.export_vlm_target_adapter_proposals import load_checkpoint


def fake_torch():
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        if len(calls) == 1:
            raise pickle.UnpicklingError("legacy config object")
        return {"ema_state_dict": {"weight": 1}}

    return SimpleNamespace(load=load), calls


def test_legacy_pickle_fallback_requires_verified_baseline():
    torch, calls = fake_torch()

    with pytest.raises(RuntimeError, match="completed matched baseline"):
        load_checkpoint(torch, "checkpoint.pt", "cuda:0")

    assert len(calls) == 1
    assert calls[0][1]["weights_only"] is True


def test_verified_baseline_uses_explicit_legacy_pickle_fallback():
    torch, calls = fake_torch()

    checkpoint = load_checkpoint(
        torch, "checkpoint.pt", "cuda:0", trusted_legacy_pickle=True
    )

    assert checkpoint["ema_state_dict"]["weight"] == 1
    assert [call[1]["weights_only"] for call in calls] == [True, False]
