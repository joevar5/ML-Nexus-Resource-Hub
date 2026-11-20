# Ex 05: A Real PyTorch Extension with Autograd

**Goal:** package your best Ex 02 kernel as an installable extension, wire
autograd, and train a tiny model with it.
**Time:** ~4 h. **Needs:** NVIDIA GPU, your Ex 02 kernel.

## Steps

1. Layout: `my_ops/matmul.cu`, `my_ops/bindings.cpp`, `setup.py` (from Lecture 03),
   `pip install -e .`. Use `AT_DISPATCH_FLOATING_TYPES` only if you template the kernel.
2. Autograd wrapper:

```python
class MyMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return my_ops.matmul(a.contiguous(), b.contiguous())
    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        # dA = g @ Bᵀ,  dB = Aᵀ @ g  — call your own kernel; .contiguous() the transposes
        return my_ops.matmul(g.contiguous(), b.t().contiguous()), \
               my_ops.matmul(a.t().contiguous(), g.contiguous())
```

3. Check gradients with `torch.autograd.gradcheck` on small **double** inputs
   (template your kernel for `double`, or gradcheck a float version with
   `atol=1e-2`) and compare to `torch.matmul` gradients on a 512×512 case.
4. Replace the matmul in a 2-layer MLP (`x @ W1 → relu → @ W2`) on random
   data; train 200 steps with both implementations from the same seed.

## Pass criteria

- `gradcheck` passes; max |grad difference| vs `torch.matmul` < 1e-3 (FP32).
- Loss curves of the two runs overlap (plot both; small last-digit drift is
  expected — say why, with Lecture 03's non-associativity point).
- A timing row: forward + backward step time for yours vs `torch.matmul`.
