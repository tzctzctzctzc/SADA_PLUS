import numpy as np

from tools.evaluate_qwen3_vl_proposal_features import normalize, summarize_frames


def test_summarize_frames_separates_static_and_transition_components():
    frames = np.asarray([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    static, transition, combined = summarize_frames(frames)
    assert static.shape == (2,)
    assert transition.shape == (2,)
    assert combined.shape == (4,)
    assert transition[0] < 0.0 < transition[1]
    assert np.isclose(np.linalg.norm(static), 1.0)
    assert np.isclose(np.linalg.norm(transition), 1.0)


def test_normalize_keeps_zero_vectors_finite():
    output = normalize(np.zeros((2, 3), dtype=np.float32))
    assert np.isfinite(output).all()
    assert np.all(output == 0.0)


def test_single_frame_has_no_artificial_transition():
    static, transition, combined = summarize_frames([[3.0, 4.0]])
    assert np.allclose(static, [0.6, 0.8])
    assert np.all(transition == 0.0)
    assert combined.shape == (4,)
