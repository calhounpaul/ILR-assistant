# ILR Language Tutor & Assessment Assistant

> *A compact, offline‑friendly Arabic comprehension tutor that meets Interagency Language Roundtable (ILR) assessment standards without reaching for cloud APIs.*

---

## Motivation

During a recent contract gig interview cycle I was asked, *“How would you build an ILR assessment assistant with modern LLMs?”* Rather than sketch a slide, I delivered a running prototype before the follow‑up call. The repo you are reading stems from that exercise.

Why go further than the brief?

1. **Prove locality matters.** A 4‑billion‑parameter model running on a Jetson is often *faster,* *cheaper,* and *safer* than shipping tokens across an API boundary.
2. **Expose hidden cost multipliers.** The same workload on GPT‑4o or Claude‑3.5 would have cost **100×** or more.
3. **Clarify persistent misconceptions.** In the past few months I've discovered that even technically literate panels at prestigious US R&D labs often don't understand that DeepSeek weights hosted in the US are not a latent sleeper‑agent risk. The weights can't phone home to China.

---

## How It Works

### 1 · Data Generation

* Queried **DeepSeek‑V3** (cheapest frontier LLM on hyperbolic.xyz, US‑hosted) to convert expert‑scored ILR PDFs into \~12 k synthetic dialogue pairs.

### 2 · Fine‑Tuning

* Base model: **Qwen3‑4B**.
* Technique: LoRA adapters via **bitsandbytes** (4‑bit).
* Hardware: Single RTX 3090, \~6 hours total.
