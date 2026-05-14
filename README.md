# SFED: SLMs For Edge Devices
- Edge Devices used: Jetson Orin Nano Super (NVIDIA), Raspberry Pi5
- The goal of this project is to experiment with hardware bottlenecks and quantization efficiency when deploying Small Language Models (SLMs) in resource-constrained edge computing environments.

## Tech Stack
Python, llama.cpp, psutil, Numpy

## File Structure
```bash
├── data (results)
│   └── ---.csv * 18
├── eval
│   ├── gemma_jetson.py
│   ├── gemma_pi.py
│   ├── llama_jetson.py
│   ├── llama_pi.py
│   ├── qwen_jetson.py
│   └── qwen_pi.py
├── plot
│   ├── for-edge-plot.ipynb
│   ├── max_peak_memory.png
│   ├── tps.png
│   └── ttft.png
└── README.md
```

## Models, Datasets, and Metrics used
**Models** * Quantization(Q5_K_M, Q4_K_M, Q3_K_M)
- Meta: [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- Google: [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- Alibaba: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

**Datasets**
- Knowledge: [MMLU](https://huggingface.co/datasets/cais/mmlu/viewer/abstract_algebra/test?row=1)
- Reasoning: [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc)
- Mathematics: [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- Programming: [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval)

**Metrics**
- Memory Usage
- Prefill TPS
- Decode TPS
- TTFT

## Experimental Results
**The Paradox of Extreme Low-Bit Quantization (Dequantization Overhead)**
* We discovered a performance inversion where the Q3_K_M quantization level, despite having the smallest model size, actually caused a drop in computational throughput (TPS). 
* This occurs because 3-bit data crosses memory byte boundaries, forcing complex bit-masking and shifting operations when loading data into registers.
* Consequently, we confirmed that this dequantization computation overhead completely overwhelms the benefits of memory bandwidth reduction, acting as a major system bottleneck.

**Heterogeneous Hardware Bottlenecks by Processing Stage (Prefill vs. Decode)**
* **Prefill Stage (GPU Dominance):** Because a large volume of prompt tokens must be processed simultaneously, the GPU (Jetson), which is optimized for parallel computing, significantly outperformed the CPU (Raspberry Pi 5)
* **Decode Stage (CPU Resilience):** The decoding process, which generates tokens sequentially, is dominated by **memory bandwidth** rather than pure compute power. Therefore, we demonstrated that a pure CPU environment with a simpler memory architecture suffers less latency loss from sequential loads, successfully maintaining stable performance.

**Critical Storage I/O Bottleneck (Cold Start)**
* We observed severe latency (up to 33 seconds) during the initial loading of model weights from storage to RAM upon an inference request.
* For on-device systems requiring real-time interaction, implementing a daemon-based architecture that keeps the weights constantly resident in memory is essential.

## Conclusion - Concise
**The Sweet Spot is Q4_K_M quantization**.
- Memory bandwidth defense
- Practical computational throughput (TPS)

**Qwen3-1.7B** demonstrated the most outstanding deployment efficiency in resource-constrained edge environments.
