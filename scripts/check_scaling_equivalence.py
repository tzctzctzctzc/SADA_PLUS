import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(17)
n, d, layers = 9, 24, 8
x = rng.normal(size=(n, d))
indices = [1, 4, 7]
weights = [tuple(rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(6)) for _ in range(layers)]
def norm(x, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
def sublayers(h, w, gate, eps):
    q, k, v, o, a, b = w
    z = norm(h, eps)
    scores = (z @ q) @ (z @ k).T / np.sqrt(d)
    scores[np.triu_indices(n, 1)] = -np.inf
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    h = h + gate * ((p @ (z @ v)) @ o)
    h = h + gate * (np.tanh(norm(h, eps) @ a) @ b)
    return h
def run(scaled, eps):
    h = x.copy()
    scales = np.ones((n, 1))
    for i, w in enumerate(weights):
        gate = np.ones((n, 1)) if scaled else 1 / scales
        h = sublayers(h, w, gate, eps)
        if i < 4:
            scales[indices] *= 8
            if scaled:
                h[indices] *= 8
    return norm(h, eps)
results = []
for eps in [0.0, 1e-6]:
    a, b = run(True, eps), run(False, eps)
    results.append({"eps": eps, "max_abs_error": float(np.max(np.abs(a-b))),
                    "relative_l2_error": float(np.linalg.norm(a-b)/np.linalg.norm(a))})
assert results[0]["max_abs_error"] < 1e-10
Path("runs/scaling_equivalence_toy.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
