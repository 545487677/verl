import os
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path
os.system('pip install EFGs')
import os
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
from EFGs import mol2frag  # 你需要提前将其导入或定义
from model import APIModel
import re
import asyncio

import itertools
import asyncio
from openai import AsyncAzureOpenAI
import itertools
import asyncio
from openai import AsyncAzureOpenAI
import time
from tqdm.asyncio import tqdm
CONCURRENT_REQUEST_LIMIT = 15
import os
from dotenv import load_dotenv
load_dotenv()
import tiktoken

MODEL_CONFIG = {
    "GPT4o-main": {
        "api_base": os.environ.get("GPT4O_MAIN_API_BASE"),
        "api_key": os.environ.get("GPT4O_MAIN_API_KEY"),
        "model": "gpt-4o",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-mini": {
        "api_base": os.environ.get("GPT4O_MINI_API_BASE"),
        "api_key": os.environ.get("GPT4O_MINI_API_KEY"),
        "model": "gpt-4o-mini",
        "api_version": "2024-03-01-preview"
    },
   "GPT4o-sunquan": {
        "api_base": os.environ.get("GPT4O_SUNQUAN_API_BASE"),
        "api_key": os.environ.get("GPT4O_SUNQUAN_API_KEY"),
        "model": "gpt-4o",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-mini-sunquan": {
        "api_base": os.environ.get("GPT4O_SUNQUAN_API_BASE"),
        "api_key": os.environ.get("GPT4O_SUNQUAN_API_KEY"),
        "model": "gpt-4o-mini",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-zhangfei": {
        "api_base": os.environ.get("GPT4O_ZHANGFEI_API_BASE"),
        "api_key": os.environ.get("GPT4O_ZHANGFEI_API_KEY"),
        "model": "gpt-4o",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-mini-zhangfei": {
        "api_base": os.environ.get("GPT4O_ZHANGFEI_API_BASE"),
        "api_key": os.environ.get("GPT4O_ZHANGFEI_API_KEY"),
        "model": "gpt-4o-mini",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-athena5": {
        "api_base": os.environ.get("GPT4O_ATHENA5_API_BASE"),
        "api_key": os.environ.get("GPT4O_ATHENA5_API_KEY"),
        "model": "gpt-4o",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-mini-athena5": {
        "api_base": os.environ.get("GPT4O_ATHENA5_API_BASE"),
        "api_key": os.environ.get("GPT4O_ATHENA5_API_KEY"),
        "model": "gpt-4o-mini",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-athena6": {
        "api_base": os.environ.get("GPT4O_ATHENA6_API_BASE"),
        "api_key": os.environ.get("GPT4O_ATHENA6_API_KEY"),
        "model": "gpt-4o",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-mini-athena6": {
        "api_base": os.environ.get("GPT4O_ATHENA6_API_BASE"),
        "api_key": os.environ.get("GPT4O_ATHENA6_API_KEY"),
        "model": "gpt-4o-mini",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-athena7": {
        "api_base": os.environ.get("GPT4O_ATHENA7_API_BASE"),
        "api_key": os.environ.get("GPT4O_ATHENA7_API_KEY"),
        "model": "gpt-4o",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-mini-athena7": {
        "api_base": os.environ.get("GPT4O_ATHENA7_API_BASE"),
        "api_key": os.environ.get("GPT4O_ATHENA7_API_KEY"),
        "model": "gpt-4o-mini",
        "api_version": "2024-03-01-preview"
    }

}

class APIModel:
    def __init__(self):
        self.model_pool = self.init_model_pool()
        self.total_tokens = 0  # 初始化总 tokens 数量
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)


    def count_tokens(self, text):
        if not isinstance(text, str):  # 检查是否是字符串
            text = ""  # 如果不是字符串，将其设置为空字符串
        return len(self.tokenizer.encode(text))

    # 初始化模型池
    def init_model_pool(self):
        clients = []
        for config in MODEL_CONFIG.values():
            client = AsyncAzureOpenAI(
                azure_endpoint=config["api_base"],
                api_key=config["api_key"],
                api_version=config["api_version"]
            )
            clients.append((client, config["model"]))
        return itertools.cycle(clients)  # 循环池

    # 异步请求，自动从池中获取模型
    async def request_with_fallback(
        self, 
        text, 
        temperature=0.7, 
        max_retries=3, 
        stream=False
    ):
        attempts = 0
        while attempts < max_retries:
            client, model = next(self.model_pool)  # 从池中获取下一个模型
            print(f"当前调用的模型: {model}")  # 打印出正在使用的模型名称
            async with self.semaphore:
                try:
                    start_time = time.time()
                    # 执行异步请求
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": text}],
                        temperature=temperature,
                        stream=stream,
                    )
                    elapsed_time = time.time() - start_time
                    # print(f"请求耗时: {elapsed_time:.2f} 秒")  # 打印耗时
                    return response if stream else response.choices[0].message.content
                except Exception as e:
                    print(f"Request failed (attempt {attempts + 1}/{max_retries}): {e}")
                    attempts += 1
                    await asyncio.sleep(2)
        raise Exception("All models failed after maximum retries.")


    # 单条消息请求
    # async def chat(self, text, temperature=1):
    #     response = await self.request_with_fallback(text, temperature=temperature)
    #     return response
    async def chat(self, text, temperature=1):
        tokens = self.count_tokens(text)  # 统计当前输入的 tokens 数量
        self.total_tokens += tokens  # 累计 tokens 数量
        response = await self.request_with_fallback(text, temperature=temperature)
        while not response:  # 如果 response 为空，继续调用直到获取有效值
            response = await self.request_with_fallback(text, temperature=temperature)

        output_tokens = self.count_tokens(response)
        self.total_tokens += output_tokens 
        return response

    # 批量请求
    async def batch_chat(self, text_batch, temperature=1):
        tasks = [self.chat(text, temperature=temperature) for text in text_batch]
        responses = []
        async for response in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            responses.append(await response)
        return responses

def extract_solution(solution_str: str) -> str:
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        return matches[-1].group(1).strip()
    return None



def is_valid_smiles(smiles: str) -> bool:
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except:
        return False

def get_structural_info(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        num_rings = mol.GetRingInfo().NumRings()
        mol_weight = Descriptors.MolWt(mol)
        return "\n".join([
            f"1. The molecule has {num_rings} ring(s).",
            f"2. The molecular weight is approximately {mol_weight:.2f} g/mol.",
        ])
    except:
        return ""

def get_fragments(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        nonCHs, CHs = mol2frag(mol)
        fragments = nonCHs + CHs
        return "".join([f"<|{frag}|>" for frag in fragments])
    except:
        return ""

def build_gpt4_prompt(desc, struct_info, frag_info, groundtruth_smi) -> str:
    prompt = (
        "You are a professional biochemist designing molecular structures.\n"
        "Given the molecular **description**, basic **structural information**, and identified **fragments**. Your goal is **not to re-predict** the SMILES, but to **generate a logical, chemically sound reasoning chain** that explains how one could deduce or construct this structure based on the given information."
        f"Description: {desc.strip()}\n"
        f"Structural Info:\n{struct_info.strip()}\n"
        f"Fragments:\n{frag_info.strip()}\n"
        "please provide a step-by-step **molecular reasoning chain** that explains how you would reconstruct or deduce the molecular structure.\n\n"
        f"Let's think step by step and return the final answer in <answer> {groundtruth_smi} </answer> tags, "
    )
    return prompt

async def process_json(json_path: str, out_path: str, split: str = "train"):
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    records = []
    seen = set()
    success_count = 0  # 记录成功样本数
    api_model = APIModel()
    for item in tqdm(raw_data):
        if success_count >= 8000:
            break

        smiles = item.get("smiles", "")
        desc = item.get("description", "")
        if not is_valid_smiles(smiles) or smiles in seen:
            continue
        seen.add(smiles)

        struct_info = get_structural_info(smiles)
        frag_info = get_fragments(smiles)
        prompt = build_gpt4_prompt(desc, struct_info, frag_info, smiles)

        try:
            cot_answer = await api_model.chat(prompt, temperature=1.0)
            pred = extract_solution(cot_answer)
            if pred != smiles:
                continue  # 如果模型预测不等于目标 SMILES，就跳过
        except Exception as e:
            print(f"⚠️ Error processing SMILES: {smiles}\n{e}")
            continue

        records.append({
            "split": split,
            "description": desc,
            "smiles": smiles,
            "structure_info": struct_info,
            "frag_info": frag_info,
            "gpt4_prompt": prompt,
            "cot_answer": cot_answer
        })
        success_count += 1

    with open(out_path, "w") as f:
        for r in records:
            json.dump(r, f)
            f.write("\n")
    print(f"✅ Saved {len(records)} samples to {out_path}")

async def main():
    await process_json(
        json_path="/vepfs/fs_projects/FunMG/LLM/dataset/datasets--GreatCaptainNemo--HME_dataset/snapshots/934c6a76f50e1f90eb83abdbc8b5366dde00639e/desc2mol_train.json",
        out_path="/vepfs/fs_projects/FunMG/LLM/dataset/mol_grpo/desc2mol_sft_cot_parquet/train_for_gpt4_prompt_max.jsonl",
        split="train"
    )
    # await process_json(
    #     json_path="/vepfs/fs_projects/FunMG/LLM/dataset/datasets--GreatCaptainNemo--HME_dataset/snapshots/934c6a76f50e1f90eb83abdbc8b5366dde00639e/desc2mol_test.json",
    #     out_path="/vepfs/fs_projects/FunMG/LLM/dataset/mol_grpo/desc2mol_sft_cot_parquet/test_for_gpt4_prompt.jsonl",
    #     split="test"
    # )


if __name__ == "__main__":
    asyncio.run(main())