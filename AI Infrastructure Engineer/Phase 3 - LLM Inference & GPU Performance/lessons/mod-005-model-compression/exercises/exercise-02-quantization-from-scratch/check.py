"""Grades starter.py. Prints PASS when every check holds."""
import sys
import numpy as np
import starter as s

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, size=(512, 1024)).astype(np.float32)
X = rng.normal(0, 1.0, size=(128, 1024)).astype(np.float32)
OUTLIERS = [3, 77, 500, 901]
X[:, OUTLIERS] *= 60.0           # a few huge activation channels, as in real LLMs
Y = X @ W.T

checks = []
def check(name, fn):
    checks.append((name, fn))

def exact_small():
    x = np.array([[1.0, -2.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
    q, sc = s.quantize(x, 8)
    assert np.allclose(sc, 2.0 / 127), f"per-tensor scale {sc}"
    assert q.tolist() == [[64, -127, 32], [0, 0, 0]], f"q = {q.tolist()}"
    q, sc = s.quantize(x, 8, per_row=True)
    assert sc.shape == (2, 1), f"per-row scale shape {sc.shape}"
    assert np.allclose(sc[:, 0], [2.0 / 127, 1.0]), f"per-row scales {sc[:, 0]} (zero row -> scale 1)"
    assert np.allclose(s.dequantize(q, sc), s.fake_quant(x, 8, per_row=True))
check("exact values on a small tensor", exact_small)

def int4_range():
    q, _ = s.quantize(W, 4, per_row=True)
    assert q.min() >= -7 and q.max() <= 7, f"int4 q out of [-7, 7]: {q.min()}..{q.max()}"
check("int4 values stay in [-7, 7]", int4_range)

def per_row_beats_tensor():
    W2 = W.copy(); W2[0] *= 20      # one loud output channel
    e_t = s.relative_error(W2, s.fake_quant(W2, 4))
    e_r = s.relative_error(W2, s.fake_quant(W2, 4, per_row=True))
    assert e_r < 0.5 * e_t, f"per-row int4 err {e_r:.3f} should be < half of per-tensor {e_t:.3f}"
check("per-channel scales beat per-tensor when one channel is loud", per_row_beats_tensor)

def groupwise_beats_row():
    W3 = W.copy(); W3[:, ::64] *= 8   # scattered large weights inside rows
    e_r = s.relative_error(W3, s.fake_quant(W3, 4, per_row=True))
    e_g = s.relative_error(W3, s.fake_quant_groupwise(W3, 4, 128))
    assert e_g < e_r, f"group-128 err {e_g:.3f} should beat per-row {e_r:.3f}"
check("group-wise int4 beats per-row int4", groupwise_beats_row)

def int8_weight_only_is_tiny():
    e = s.relative_error(Y, X @ s.fake_quant(W, 8, per_row=True).T)
    assert e < 0.015, f"W8 (per-channel) output error {e:.4f} should be < 1.5%"
check("int8 per-channel weight-only error < 1.5%", int8_weight_only_is_tiny)

def smoothquant_helps():
    naive = s.fake_quant(X, 8) @ s.fake_quant(W, 8, per_row=True).T
    sc = s.smoothquant_scales(np.abs(X).max(0), np.abs(W).max(0), alpha=0.5)
    assert sc.shape == (1024,), f"scales shape {sc.shape}"
    Xs, Ws = X / sc, W * sc
    assert np.allclose(Xs @ Ws.T, Y, rtol=1e-3, atol=1e-3), "smoothing must not change X @ W.T"
    smooth = s.fake_quant(Xs, 8) @ s.fake_quant(Ws, 8, per_row=True).T
    e_n, e_s = s.relative_error(Y, naive), s.relative_error(Y, smooth)
    assert e_s < 0.5 * e_n, f"W8A8 error after SmoothQuant {e_s:.4f} should be < half of naive {e_n:.4f}"
check("SmoothQuant cuts W8A8 error by > 2x on outlier activations", smoothquant_helps)

fails = []
for name, fn in checks:
    try:
        fn()
    except NotImplementedError:
        fails.append(f"  {name}: not implemented")
    except AssertionError as e:
        fails.append(f"  {name}: {e}")
    except Exception as e:  # noqa: BLE001 — surface any student error
        fails.append(f"  {name}: raised {type(e).__name__}: {e}")
if fails:
    print(f"FAIL ({len(fails)}/{len(checks)})"); print("\n".join(fails)); sys.exit(1)
print(f"PASS ({len(checks)}/{len(checks)})")
