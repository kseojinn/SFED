import time
import os
import csv
import psutil
import numpy as np
import multiprocessing
from datasets import load_dataset, concatenate_datasets
from llama_cpp import Llama

MODEL_DIR = "/home/jin/win_models"
LLAMA_MODELS = [
    "Llama-3.2-3B-Instruct-Q3_K_M.gguf",
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "Llama-3.2-3B-Instruct-Q5_K_M.gguf"
]
OUTPUT_FILE = "jetson_llama3.2_bench_results.csv"

# 각 조건당 3회 반복 측정
NUM_RUNS = 3 
MAX_TOKENS_LIST = [128, 256, 512]

def get_current_memory():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def load_bench_prompts(num_samples=200):
    print(f"데이터셋 5종 로드 및 프롬프트 추출 (목표: {num_samples}개).")
    prompts_dict = {}

    # 1. MMLU (CS + Math 병합)
    ds_cs = load_dataset("cais/mmlu", "college_computer_science", split="test")
    ds_math = load_dataset("cais/mmlu", "college_mathematics", split="test")
    ds_mmlu_combined = concatenate_datasets([ds_cs, ds_math])
    
    take_mmlu = min(num_samples, len(ds_mmlu_combined)) 
    ds_mmlu = ds_mmlu_combined.select(range(take_mmlu))
    prompts_dict["MMLU"] = [
        f"Question: {d['question']}\nA. {d['choices'][0]}\nB. {d['choices'][1]}\nC. {d['choices'][2]}\nD. {d['choices'][3]}\nAnswer with only the letter." 
        for d in ds_mmlu
    ]

    # 2. ARC-Challenge
    ds_arc = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    take_arc = min(num_samples, len(ds_arc))
    ds_arc = ds_arc.select(range(take_arc))
    prompts_dict["ARC"] = [
        f"Question: {d['question']}\n" + "\n".join([f"{l}. {t}" for l, t in zip(d['choices']['label'], d['choices']['text'])]) + "\nAnswer with only the letter/number."
        for d in ds_arc
    ]

    # 3. HellaSwag 
    ds_hellaswag = load_dataset("Rowan/hellaswag", split="validation")
    take_hella = min(num_samples, len(ds_hellaswag))
    ds_hellaswag = ds_hellaswag.select(range(take_hella))
    prompts_dict["HellaSwag"] = [
        f"Context: {d['ctx']}\nWhich ending makes the most sense?\nA. {d['endings'][0]}\nB. {d['endings'][1]}\nC. {d['endings'][2]}\nD. {d['endings'][3]}\nAnswer with only the letter."
        for d in ds_hellaswag
    ]

    # 4. TruthfulQA 
    ds_truthfulqa = load_dataset("truthful_qa", "generation", split="validation")
    take_truth = min(num_samples, len(ds_truthfulqa))
    ds_truthfulqa = ds_truthfulqa.select(range(take_truth))
    prompts_dict["TruthfulQA"] = [
        f"Question: {d['question']}\nAnswer concisely and truthfully:"
        for d in ds_truthfulqa
    ]

    # 5. GSM8K 
    ds_gsm = load_dataset("gsm8k", "main", split="test")
    take_gsm = min(num_samples, len(ds_gsm))
    ds_gsm = ds_gsm.select(range(take_gsm))
    prompts_dict["GSM8K"] = [
        f"Question: {d['question']}\nLet's think step by step. End with 'The answer is [number]'" 
        for d in ds_gsm
    ]

    return prompts_dict

def format_llama_prompt(prompt):
    return f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def run_performance_bench(model_path, datasets_prompts):
    model_name = os.path.basename(model_path)
    print(f"\n========================================")
    print(f"[성능 분석 시작] 모델: {model_name}")
    print(f"========================================")
    
    # [추가됨] 1. 모델 로드 시간 측정
    start_load = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1, 
        n_ctx=2048,
        verbose=False
    )
    load_time = time.perf_counter() - start_load
    print(f"  -> 모델 로드 완료: {load_time:.2f}s")
    
    def _measure_inference(prompt_text, target_max_tokens):
        prompt_formatted = format_llama_prompt(prompt_text)
        prompt_tokens = len(llm.tokenize(prompt_formatted.encode('utf-8')))
        
        start_time = time.perf_counter()
        
        output = llm(
            prompt_formatted,
            max_tokens=target_max_tokens,
            stop=["<|eot_id|>"],
            stream=True
        )
        
        first_token_time = None
        generated_text = ""
        generated_token_count = 0
        
        for chunk in output:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            generated_text += chunk['choices'][0]['text']
            generated_token_count += 1
            
        end_time = time.perf_counter()
        
        ttft = max(first_token_time - start_time, 1e-9) if first_token_time else 1e-9
        decode_time = max(end_time - first_token_time, 1e-9) if first_token_time else 1e-9
        
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

    # [추가됨] 2. 웜업(Warm-up) 수행 및 콜드 스타트 TTFT 측정
    print("  -> 웜업(Warm-up) 및 콜드 스타트 TTFT 측정 중...")
    warmup_res = _measure_inference("Hello, testing cold start.", target_max_tokens=16)
    cold_start_ttft = warmup_res['ttft']
    print(f"  -> 콜드 스타트 TTFT 확인: {cold_start_ttft:.3f}s\n")

    dataset_metrics = []

    for target_max_tokens in MAX_TOKENS_LIST:
        print(f"\n============== [Max Tokens: {target_max_tokens}] ==============")
        
        for dataset_name, prompts in datasets_prompts.items():
            print(f"\n  [{dataset_name}] {NUM_RUNS}회 반복 측정 진행 (샘플 수: {len(prompts)}개)")
            
            run_ttfts = []
            run_prefill_tps = [] 
            run_tps = []         
            run_mems = []

            for run in range(NUM_RUNS):
                print(f"    -> Run {run + 1}/{NUM_RUNS} 시작...")
                run_avg_ttft_list = []
                run_avg_prefill_tps_list = [] 
                run_avg_tps_list = []
                peak_mem = 0
                
                for idx, prompt in enumerate(prompts):
                    res = _measure_inference(prompt, target_max_tokens)
                    run_avg_ttft_list.append(res['ttft'])
                    run_avg_prefill_tps_list.append(res['prefill_tps']) 
                    run_avg_tps_list.append(res['decode_tps'])
                    
                    if res['current_mem'] > peak_mem:
                        peak_mem = res['current_mem']
                    
                    if (idx + 1) % 10 == 0 or (idx + 1) == len(prompts):
                        print(f"      진행 상황: {idx + 1} / {len(prompts)} 완료 (최근 측정 - TTFT: {res['ttft']:.2f}s, Decode TPS: {res['decode_tps']:.2f})")
                
                run_ttfts.append(np.mean(run_avg_ttft_list))
                run_prefill_tps.append(np.mean(run_avg_prefill_tps_list)) 
                run_tps.append(np.mean(run_avg_tps_list))
                run_mems.append(peak_mem)
                print(f"       => Run {run + 1} 완료 | 평균 TTFT: {run_ttfts[-1]:.3f}s | 평균 Prefill TPS: {run_prefill_tps[-1]:.2f} | 평균 Decode TPS: {run_tps[-1]:.2f}")

            # [추가됨] 3. 결과 CSV에 콜드 스타트 지표(Model_Load_Time, Cold_Start_TTFT) 추가 반영
            metrics = {
                "Model": model_name,
                "Dataset": dataset_name,
                "Max_Tokens": target_max_tokens, 
                "Runs": NUM_RUNS,
                "Model_Load_Time (s)": round(load_time, 2),        # 메모리 적재 시간
                "Cold_Start_TTFT (s)": round(cold_start_ttft, 4),  # 첫 번째 토큰 생성 시간 (콜드 스타트)
                "TTFT_Mean (s)": round(np.mean(run_ttfts), 4),     # 웜(Warm) 상태의 평균 TTFT
                "TTFT_Std (s)": round(np.std(run_ttfts), 4),
                "Prefill_TPS_Mean": round(np.mean(run_prefill_tps), 4), 
                "Prefill_TPS_Std": round(np.std(run_prefill_tps), 4),   
                "Decode_TPS_Mean": round(np.mean(run_tps), 4),
                "Decode_TPS_Std": round(np.std(run_tps), 4),
                "Max_Peak_Mem (MB)": round(max(run_mems), 1)
            }
            
            dataset_metrics.append(metrics)
            print(f"  => [{dataset_name} | Max: {target_max_tokens}] 최종 결과: Decode TPS {metrics['Decode_TPS_Mean']} ± {metrics['Decode_TPS_Std']}")

            file_exists = os.path.isfile(OUTPUT_FILE)
            with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(metrics)

def benchmark_worker(model_path, datasets_prompts):
    run_performance_bench(model_path, datasets_prompts)

if __name__ == "__main__":
    datasets_prompts = load_bench_prompts(num_samples=200)
    
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    for model_name in LLAMA_MODELS:
        full_path = os.path.join(MODEL_DIR, model_name)
        if os.path.exists(full_path):
            print(f"\n 서브프로세스 할당: '{model_name}'")
            p = multiprocessing.Process(target=benchmark_worker, args=(full_path, datasets_prompts))
            p.start()
            p.join()
            print(f"\n서브프로세스 종료. 메모리 반환. 15초 대기.")
            time.sleep(15)
        else:
            print(f"[에러] File Not Found {full_path}")

    print(f"\n벤치마크 완료. 결과가 '{OUTPUT_FILE}'에 저장되었습니다.")
