from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import re

# ----------- GSM8k 专用提取答案函数 -------------
def extract_solution(solution_str, method="strict"):
    if method == "strict":
        match = re.search(r"#### (\-?[0-9\.,]+)", solution_str)
        return match.group(1).replace(",", "") if match else None
    elif method == "flexible":
        numbers = re.findall(r"(\-?[0-9\.,]+)", solution_str)
        return numbers[-1].replace(",", "") if numbers else None

def compute_score(pred_str, gt, method="strict"):
    pred = extract_solution(pred_str, method)
    return int(pred == gt) if pred else 0

# ----------- 模型路径和数据路径 -------------
model_path = "/vepfs/fs_ckps/guojianz/llm/verl_exp/verl_grpo_example_gsm8k_reimplement/qwen2_7b_function_rm/merged_hf_model"
data_path = "/vepfs/fs_projects/FunMG/LLM/dataset/gsm8k/test.parquet"

# ----------- 加载模型和 tokenizer -------------
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    trust_remote_code=True,
    max_model_len=2048,               # ✅ 显式指定可接受的最大上下文长度
    gpu_memory_utilization=0.90       # ✅ 可提高占用率（默认 0.85）
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
)

# ----------- 读取 GSM8k 数据 -------------
df = pd.read_parquet(data_path)
results = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["prompt"][0]["content"]

    gt = row["reward_model"]["ground_truth"]


    # 直接使用 LLM 推理（同步）
    output = llm.generate([prompt], sampling_params=sampling_params)[0].outputs[0].text
    score = compute_score(output, gt)

    results.append({
        "index": i,
        "prompt": prompt,
        "gt": gt,
        "output": output,
        "score": score
    })

# ----------- 保存结果并打印准确率 -------------
df_out = pd.DataFrame(results)
df_out.to_csv("gsm8k_vllm_eval.csv", index=False)
print("✅ 推理完成")
print(f"📊 Pass@1 accuracy: {df_out['score'].mean():.4f}")
