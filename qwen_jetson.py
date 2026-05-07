import time
import os
import csv
import psutil
import numpy as np
import multiprocessing
from datasets import load_dataset
from llama_cpp import Llama

MODEL_DIR = "/home/jin/win_models"
QWEN_MODELS = [
    "Qwen3-1.7B-Q5_K_M.gguf",
    "Qwen3-1.7B-Q4_K_M.gguf",
    "Qwen3-1.7B-Q3_K_M.gguf"
]
OUTPUT_FILE = "qwen_2048_128.csv"

def get_current_memory():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def load_bench_prompts(num_samples=20):
    print("데이터셋 4종 로드 및 프롬프트 추출.")
    prompts_dict = {}

    ds_mmlu = load_dataset("cais/mmlu", "college_computer_science", split="test").select(range(num_samples))
    prompts_dict["MMLU"] = [
        f"Question: {d['question']}\nA. {d['choices'][0]}\nB. {d['choices'][1]}\nC. {d['choices'][2]}\nD. {d['choices'][3]}\nAnswer with only the letter." 
        for d in ds_mmlu
    ]

    ds_arc = load_dataset("ai2_arc", "ARC-Challenge", split="test").select(range(num_samples))
    prompts_dict["ARC"] = [
        f"Question: {d['question']}\n" + "\n".join([f"{l}. {t}" for l, t in zip(d['choices']['label'], d['choices']['text'])]) + "\nAnswer with only the letter/number."
        for d in ds_arc
    ]

    ds_gsm = load_dataset("gsm8k", "main", split="test").select(range(num_samples))
    prompts_dict["GSM8K"] = [
        f"Question: {d['question']}\nLet's think step by step. End with 'The answer is [number]'" 
        for d in ds_gsm
    ]

    ds_code = load_dataset("openai_humaneval", split="test").select(range(num_samples))
    prompts_dict["HumanEval"] = [
        f"Complete the Python function:\n```python\n{d['prompt']}\n```" 
        for d in ds_code
    ]

    return prompts_dict

def format_qwen_prompt(prompt):
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def run_performance_bench(model_path, datasets_prompts):
    model_name = os.path.basename(model_path)
    print(f"\n========================================")
    print(f"[성능 분석 시작] 모델: {model_name}")
    print(f"========================================")
    
    start_load = time.perf_counter()
    
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1, 
        n_ctx=2048,
        verbose=False
    )
    load_time = time.perf_counter() - start_load
    
    def _measure_inference(prompt_text, max_tokens=128):
        prompt_formatted = format_qwen_prompt(prompt_text)
        prompt_tokens = len(llm.tokenize(prompt_formatted.encode('utf-8')))
        
        start_time = time.perf_counter()
        
        output = llm(
            prompt_formatted,
            max_tokens=max_tokens,
            stop=["<|im_end|>"], 
            stream=True
        )
        
        first_token_time = None
        generated_text = ""
        
        for chunk in output:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            generated_text += chunk['choices'][0]['text']
            
        end_time = time.perf_counter()
        
        generated_token_count = len(llm.tokenize(generated_text.encode('utf-8')))
        
        ttft = first_token_time - start_time if first_token_time else 0
        decode_time = end_time - first_token_time if first_token_time else 0
        
        prefill_tps = prompt_tokens / ttft if ttft > 0 else 0
        decode_tps = (generated_token_count - 1) / decode_time if decode_time > 0 and generated_token_count > 1 else 0
        
        return {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_token_count,
            "ttft": ttft,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "current_mem": get_current_memory() 
        }

    print("  [워밍업] GPU 초기화 및 콜드 스타트 지연 측정.")
    cold_result = _measure_inference("Hello. Reply with 'Hi' only.", max_tokens=10)
    print(f"  -> 콜드 스타트 TTFT: {cold_result['ttft']:.3f}s (Decode: {cold_result['decode_tps']:.2f} TPS)\n")

    dataset_metrics = []
    dataset_keys = list(datasets_prompts.keys())

    for idx, (dataset_name, prompts) in enumerate(datasets_prompts.items()):
        print(f"  [{dataset_name}] 테스트 진행 중 (총 {len(prompts)}개)...")
        results = []
        
        dataset_peak_mem = 0
        
        for i, prompt in enumerate(prompts):
            res = _measure_inference(prompt)
            results.append(res)
            
            if res['current_mem'] > dataset_peak_mem:
                dataset_peak_mem = res['current_mem']
                
            print(f"    - {i+1}번 | 프롬프트: {res['prompt_tokens']} | 생성: {res['generated_tokens']} | TTFT: {res['ttft']:.3f}s | TPS: {res['decode_tps']:.2f} | 현재 메모리: {res['current_mem']:.1f} MB") 
            
        all_tokens_str = ", ".join([str(r['prompt_tokens']) for r in results])
        all_ttft_str = ", ".join([f"{r['ttft']:.3f}" for r in results])
        all_tps_str = ", ".join([f"{r['decode_tps']:.2f}" for r in results])
            
        metrics = {
            "Model": model_name,
            "Dataset": dataset_name,
            "Total Prompt Tokens": sum([r['prompt_tokens'] for r in results]),
            "Avg Prompt Tokens": np.mean([r['prompt_tokens'] for r in results]),
            "All Prompt Tokens": all_tokens_str,
            "Cold Start TTFT (s)": cold_result['ttft'] if dataset_name == "MMLU" else "-", 
            "Avg TTFT (s)": np.mean([r['ttft'] for r in results]),
            "All TTFTs (s)": all_ttft_str,
            "Avg Prefill TPS": np.mean([r['prefill_tps'] for r in results]),
            "Avg Decode TPS": np.mean([r['decode_tps'] for r in results]),
            "All Decode TPS": all_tps_str,
            "Dataset Peak Memory (MB)": dataset_peak_mem
        }
        dataset_metrics.append(metrics)
        print(f"  => [{dataset_name}] 평균 Decode TPS: {metrics['Avg Decode TPS']:.2f} | 📈 데이터셋 피크 메모리: {dataset_peak_mem:.1f} MB")

        file_exists = os.path.isfile(OUTPUT_FILE)
        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
            
        if idx < len(dataset_keys) - 1:
            cooldown_seconds = 15
            print(f"\n  {cooldown_seconds}초 대기.")
            time.sleep(cooldown_seconds)
            print("  다음 테스트 진행.\n")

def benchmark_worker(model_path, datasets_prompts):
    run_performance_bench(model_path, datasets_prompts)


if __name__ == "__main__":
    datasets_prompts = load_bench_prompts(num_samples=20)
    
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    for model_name in QWEN_MODELS:
        full_path = os.path.join(MODEL_DIR, model_name)
        if os.path.exists(full_path):
            print(f"\n서브프로세스 할당: '{model_name}'")
            
            p = multiprocessing.Process(target=benchmark_worker, args=(full_path, datasets_prompts))
            p.start()
            p.join()
            
            print(f"\n서브프로세스 종료. 메모리 반환. 15초 대기.")
            time.sleep(15)
        else:
            print(f"[에러] File Not Found {full_path}")

    print(f"\n벤치마크 결과 '{OUTPUT_FILE}' 저장.")
