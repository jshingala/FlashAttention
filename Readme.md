# FlashAttention from Scratch (Triton)

A from-scratch implementation of FlashAttention as a fused Triton kernel, built up from
first principles on a single NVIDIA T4. The repo walks from a trivial copy kernel → a
numerically-stable softmax kernel → the full fused attention kernel with the
online-softmax (streaming) recurrence, then benchmarks it against naive PyTorch attention
and `torch.nn.functional.scaled_dot_product_attention` (SDPA).

**Scope (the honest version).** This is an educational, correctness-first implementation.
The attention kernel is numerically correct (matches PyTorch) and achieves FlashAttention's
defining property — **O(N) memory instead of O(N²)**. It is **not yet performance-tuned**:
throughput is currently well below SDPA because the kernel is un-autotuned (default
`num_warps`/`num_stages`, fixed 64×64 tiles, no software pipelining). Closing that gap is
the top item on the roadmap below.

---

## What's implemented

- **`copy_kernel`** — the minimal Triton kernel: program IDs, block offsets, masked
  load/store. A mechanics warm-up.
- **`softmax_kernel`** — row-wise, numerically-stable softmax (subtract the row max,
  exponentiate, normalize). One program per row; padding columns loaded as `-inf`.
- **`flash_attention_kernel`** — the headline. Fused scaled-dot-product attention that
  streams K/V in blocks and never materializes the full N×N scores matrix, using the
  online-softmax recurrence: a running max `m`, a running sum of exponentials `l`, and a
  running weighted output `acc`, all rescaled by `alpha = exp(m_old − m_new)` whenever the
  running max grows.

All three are verified against PyTorch references.

---

## Results (NVIDIA T4, fp16)

### Correctness
`flash_attention` matches naive `softmax(QKᵀ/√d)·V` to within fp16 tolerance (`corr = ok`)
at every tested size, and to a max absolute difference of ~`5e-7` in fp32.

### Memory — the headline result
Peak GPU memory attributable to a single attention call. Naive grows **quadratically** (it
allocates the full N×N scores tensor); Flash grows **linearly**.

| N (seq len) | naive peak | flash peak |
|------------:|-----------:|-----------:|
| 512   | 17.8 MB    | 1.0 MB  |
| 1024  | 69.2 MB    | 2.1 MB  |
| 2048  | 272.6 MB   | 4.2 MB  |
| 4096  | 1082.1 MB  | 8.4 MB  |
| 8192  | 4311.7 MB  | 16.8 MB |

Naive scales ~4× per doubling of N (O(N²)); Flash scales ~2× per doubling (O(N)). Beyond
~8K, naive attention OOMs on a 16 GB T4 while Flash keeps running — this divergence is the
entire reason FlashAttention exists. *(Config: B=1, H=16, D=64.)*

### Throughput — honest, pre-tuning
| N    | flash       | sdpa      | naive     |
|-----:|------------:|----------:|----------:|
| 512  | 2.376 ms    | 0.137 ms  | 0.385 ms  |
| 1024 | 8.747 ms    | 0.400 ms  | 1.086 ms  |
| 2048 | 34.778 ms   | 1.547 ms  | 4.342 ms  |
| 4096 | 137.969 ms  | 5.855 ms  | 20.085 ms |
| 8192 | 547.192 ms  | 22.204 ms | 77.542 ms |

The kernel currently sustains **~0.5 TFLOP/s** — under 1% of the T4's ~65 TFLOP/s fp16 peak —
making it roughly **7× slower than naive PyTorch** and **25× slower than SDPA**. Note that
FlashAttention does *not* reduce FLOPs (compute is still O(N²)); this is a pure
constant-factor gap: the `tl.dot`s are not yet saturating the tensor cores, and the kernel
is un-pipelined and un-autotuned. See the Roadmap.

---

## How it works

> **TODO — write this section in your own words.**
> This is the part an interviewer drills into, so it has to be yours, not borrowed. Hit
> these three beats:
>
> 1. **The problem with naive attention.** Why materializing the full N×N scores matrix is
>    the O(N²) memory blowup that FlashAttention exists to eliminate.
>
> 2. **The streaming idea.** Walk K/V in blocks; never build the full matrix; keep three
>    running quantities per query row — `m` (largest score seen so far), `l` (running sum of
>    exponentials, relative to the current `m`), and `acc` (running weighted output, also
>    relative to the current `m`).
>
> 3. **The rescale — the load-bearing idea.** When a new block arrives with a larger max
>    than the running `m`, *why* are the old `l` and `acc` now wrong, and what does
>    `alpha = exp(m_old − m_new)` do to correct them in a single multiply? Finish with: why
>    you normalize (`acc / l`) exactly once, after the loop. *(This is the `alpha` sentence —
>    write it cleanly right here.)*

---

## Limitations / known issues

- **Throughput is un-tuned** (~0.5 TFLOP/s) — the top roadmap item.
- **No causal masking** — full bidirectional attention only; not yet usable as a decoder.
- **Fixed tiles, no autotuning** — `BLOCK_M = BLOCK_N = 64`, default `num_warps`/`num_stages`.
- **Benchmark does not complete at N=16384** — at that size one (slow) kernel call takes
  ~2 s, and the harness times it 125× (warmup 25 + iters 100), so it hangs and must be
  interrupted. Reduce timing iterations at large N; once the kernel is autotuned, this
  becomes feasible and will capture the naive-OOM-vs-Flash-continues row.
- **HEAD_DIM is assumed to fit in SRAM** (tested at D=64).

---

## Roadmap

1. **Autotuning.** Wrap the kernel in `@triton.autotune`, sweeping `num_warps` (4/8),
   `num_stages` (2–4), and `BLOCK_M`/`BLOCK_N` (64/128); profile one config to confirm the
   MMAs are hitting tensor cores. Expected to move throughput from ~0.5 TFLOP/s to several
   TFLOP/s with no changes to the math.
2. **Capture the N=16384 benchmark row** (with reduced timing iterations) so the chart
   shows naive OOM while Flash continues.
3. **Causal masking.** Add `offs_m[:, None] >= offs_n[None, :]` and skip key blocks that lie
   entirely past the diagonal.
4. **Render the charts** (matplotlib) from the benchmark table — the memory curve is the
   money chart.
5. **Upstream contribution.** A small PR to vLLM / SGLang / FlashInfer.

---

## Setup & usage

Requires a CUDA GPU. Tested on an NVIDIA T4 (Google Colab) with PyTorch and Triton 3.6.0.

```bash
pip install torch triton
```

```python
# q, k, v: [B, H, N, D] on cuda
out = flash_attention(q, k, v)
```

Run the benchmark harness:

```python
run(dtype=torch.float16)   # pass torch.float32 to see the SDPA-dispatch gap
```
