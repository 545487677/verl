import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]
    
    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score


model_path = "/vepfs/fs_ckps/guojianz/llm/verl_exp/verl_grpo_example_gsm8k_reimplement/qwen2_7b_function_rm/merged_hf_model"
data_path = "/vepfs/fs_projects/FunMG/LLM/dataset/gsm8k/gsm8k/test.parquet"  # 替换成你的 parquet 测试数据路径

# Load tokenizer and model (自动多卡 + FP16 加速)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# Load test data
df = pd.read_parquet(data_path)

all_scores = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["prompt"]
    ground_truth = row["reward_model"]["ground_truth"]
    if isinstance(prompt, list):
        prompt = prompt[0]["content"]

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    gen_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # reward calculation
    reward = compute_score(solution_str=gen_answer, ground_truth=ground_truth, method="strict")
    all_scores.append({
        "index": i,
        "ground_truth": ground_truth,
        "prediction": gen_answer,
        "reward": reward,
    })

# Save results
results_df = pd.DataFrame(all_scores)
results_df.to_csv("/vepfs/fs_users/guojianz/dp_project/LLM/LLM_Finetune/Finetune_learning/verl/examples/generation/gsm8k_eval_results.csv", index=False)
print("✅ 推理完成，保存至 gsm8k_eval_results.csv")
print("📊 平均得分：", results_df["reward"].mean())
