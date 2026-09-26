# MetaX Muxi C500 GPU — LLM & Stable Diffusion Performance Testing

The primary goal of this project was to evaluate inference performance across different model sizes and configurations, focusing on **tokens per second**.

---

## Overview

During my experience in an AI supercomputing lab, I tested the **MetaX Muxi C500 GPU** using:

* **Llama 3.3 70B** — multi-GPU inference using vLLM
* **Llama 3.2 1B** — single-GPU inference using PyTorch
* **Stable Diffusion v2-1** — image generation performance

The primary performance metric for LLM inference was **tokens per second (tokens/s)**.

For Stable Diffusion, performance was measured as **seconds per generated image**.

---

## GPU Specifications

| Metric                              |           Value |
| ----------------------------------- | --------------: |
| GPU                                 | MetaX Muxi C500 |
| Memory Capacity                     |           64 GB |
| Memory Bandwidth                    |       1.8 TB/s* |
| TDP                                 |           350 W |
| FP16 Compute Performance            |     219 TFLOPS† |
| Maximum Memory Utilization Observed |             90% |
| Memory Used                         |           54 GB |
| GPU Utilization                     |             33% |
| `mx-smi` Version                    |          2.1.10 |

* Memory bandwidth reported by Muxi.
† FP16 TFLOPS based on testing performed during this project.

---

# LLM Inference Benchmarks

## Llama 3.3 70B

| Max Batch Size | Max Sequence Size | Tokens/s | Max Completion Tokens | Prompt Tokens | Output Tokens |
| -------------: | ----------------: | -------: | --------------------: | ------------: | ------------: |
|           2048 |               256 |    27.10 |                   256 |            41 |           297 |
|           2048 |               256 |    27.10 |                   512 |            41 |           405 |
|           1024 |               256 |    30.90 |                   256 |            41 |           297 |
|           1024 |               256 |    27.13 |                   512 |            41 |           507 |
|              1 |              4096 |    29.18 |                   256 |            41 |           297 |
|              1 |              4096 |    29.24 |                   512 |            41 |           553 |
|           1024 |              4096 |    29.30 |                   256 |            41 |           297 |

---

## Llama 3.2 1B

| Max Batch Size | Max Sequence Size | Tokens/s | Max Completion Tokens | Prompt Tokens | Output Tokens |
| -------------: | ----------------: | -------: | --------------------: | ------------: | ------------: |
|           2048 |               256 |    77.65 |                   512 |             6 |           513 |
|           1024 |               256 |    75.71 |                   512 |             6 |           420 |
|           1024 |               256 |    72.79 |                   256 |             6 |           257 |
|           2048 |               256 |    73.23 |                   256 |             6 |           257 |

---

# Stable Diffusion Benchmark

**Model:** Stable Diffusion v2-1

| Batch Size |    Performance |
| ---------: | -------------: |
|          1 | 3.33 sec/image |
|          8 | 2.36 sec/image |

The larger batch size improved per-image throughput from **3.33 sec/image** to **2.36 sec/image**.

### Software

* PyTorch
* vLLM
* Stable Diffusion
* Muxi GPU software stack
* `mx-smi 2.1.10`

---

# Experience Context

The reported results are **experimental benchmark results**, not universal performance specifications for the Muxi C500.

The work focused on evaluating AI workloads on Chinese GPU hardware and gaining practical experience with GPU infrastructure, LLM inference, performance measurement, and AI software stacks.
