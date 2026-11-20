# Ex 01: nsys Trace of LLM Generation — Prefill vs Decode

**Goal:** see the two phases of inference on a real timeline and find out
what limits decode. Answers question 1 and part of 13 in
[the 25 questions](../../../25-questions.md).
**Time:** ~3 h. **Needs:** NVIDIA GPU, `nsys`, `transformers`.

## Setup

```python
# gen.py
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen2.5-0.5B-Instruct"          # any ≤1B causal LM works
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
prompt = tok("Explain the roofline model. " * 40, return_tensors="pt").to("cuda")  # ~400 tokens
with torch.no_grad():
    model.generate(**prompt, max_new_tokens=8)                    # warm-up
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("generate"):
        out = model.generate(**prompt, max_new_tokens=128, do_sample=False)
```

Wrap the **first forward call** in an extra `nvtx.range("prefill")` (a forward
hook on the model that fires once works), and run:

```bash
nsys profile -t cuda,nvtx,osrt --capture-range=nvtx -p generate -o gen python gen.py
nsys stats gen.nsys-rep --report cuda_gpu_kern_sum
nsys analyze gen.nsys-rep            # gpu_gaps, gpu_time_util rules
```

## Questions to answer in `HOTSPOTS.md`

1. Prefill time vs average decode-step time (ms). How many kernels per decode step?
2. Top 3 kernels by total GPU time; for each: name, total time, % of GPU time, calls.
3. During decode, what fraction of wall time is the GPU idle (gaps)? Is the
   Python/launch thread the bottleneck? (Look at the CPU row vs GPU row.)
4. **Predict** the decode step time from the roofline: model bytes ÷ your GPU's
   bandwidth (Mod 1 Lecture 1.5 deep dive). How far is the measured step from it?

## Pass criteria

- Screenshot of the timeline with the prefill range and ≥ 3 decode steps visible.
- The four answers above, with numbers.
- A one-line hypothesis for the decode gap (expected: launch/CPU overhead for a
  small model — you will test it in Ex 03).
