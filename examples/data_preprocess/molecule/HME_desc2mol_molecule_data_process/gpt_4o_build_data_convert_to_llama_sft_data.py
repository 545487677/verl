import os
import json
import re
import time
import asyncio
import itertools
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm, asyncio as tqdm_asyncio
from rdkit import Chem
from rdkit.Chem import Descriptors
from EFGs import mol2frag  
from openai import AsyncAzureOpenAI
import tiktoken
load_dotenv()
CONCURRENT_REQUEST_LIMIT = 15
os.system('pip install EFGs')

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
        self.total_tokens = 0  
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)
        self.total_calls = 0 

    def count_tokens(self, text):
        if not isinstance(text, str):  
            text = ""  
        return len(self.tokenizer.encode(text))

    def init_model_pool(self):
        clients = []
        for config in MODEL_CONFIG.values():
            client = AsyncAzureOpenAI(
                azure_endpoint=config["api_base"],
                api_key=config["api_key"],
                api_version=config["api_version"]
            )
            clients.append((client, config["model"]))
        return itertools.cycle(clients) 

    async def request_with_fallback(
        self, 
        text, 
        temperature=0.7, 
        max_retries=3, 
        stream=False
    ):
        attempts = 0
        while attempts < max_retries:
            client, model = next(self.model_pool)  
            async with self.semaphore:
                try:
                    start_time = time.time()
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": text}],
                        temperature=temperature,
                        stream=stream,
                    )
                    elapsed_time = time.time() - start_time
                    print(f"请求耗时: {elapsed_time:.2f} 秒")  
                    return response if stream else response.choices[0].message.content
                except Exception as e:
                    print(f"Request failed (attempt {attempts + 1}/{max_retries}): {e}")
                    attempts += 1
                    await asyncio.sleep(2)
        raise Exception("All models failed after maximum retries.")

    async def chat(self, text, temperature=1):
        tokens = self.count_tokens(text)  
        self.total_tokens += tokens 
        response = await self.request_with_fallback(text, temperature=temperature)
        self.total_calls += 1
        while not response:  
            response = await self.request_with_fallback(text, temperature=temperature)

        output_tokens = self.count_tokens(response)
        self.total_tokens += output_tokens 
        return response

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

async def process_json(json_path: str, out_path: str, sft_path:str, split: str = "train"):
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    print('origin data: ', len(raw_data))
    records = []
    seen = set()
    success_count = 0  
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
                continue  
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
    print("🧾 推理统计信息：")
    print(f"✅ 总调用次数: {api_model.total_calls}")
    print(f"✅ 累计 token 数量: {api_model.total_tokens}")
    print(f"✅ 估算 token 成本（按 gpt-4o-mini，$0.0005/1k tokens）: ${api_model.total_tokens / 1000 * 0.0005:.4f}")
    print(f"✅ 估算 token 成本（按 gpt-4o，$0.005/1k tokens）: ${api_model.total_tokens / 1000 * 0.005:.4f}")
    train_dataset = []
    with open(out_path, "r") as all_f:
        for line in all_f.readlines():
            data = json.loads(line)
            trans_data = {}
            trans_data["instruction"] = "You are a professional biochemist designing molecular structures. Please generate the molecular structure (SMILES) based on the following description.\n\n Please think step by step and return the final answer in <answer> </answer> tags, for example <answer> CCCCC[C@@H]1O[C@@H]1/C=C/C(O)C/C=C\\C/C=C\\CCCC(=O)[O-] </answer>. Do not use <|...|> or other formats. The answer **must** be returned in the format: <answer> CCCCC[C@@H]1O[C@@H]1/C=C/C(O)C/C=C\\C/C=C\\CCCC(=O)[O-] </answer>."
            trans_data["input"] = data["description"]
            smiles = data['smiles']

            if '.' in smiles:
                continue

            cot_answer = data["cot_answer"]
            if re.search(r"\\u[0-9a-fA-F]{4}", cot_answer):
                continue

            if any(ord(c) > 127 for c in cot_answer):
                continue
            
            pred = extract_solution(cot_answer)
            if pred != smiles:
                print('invalid')
                continue

            trans_data["output"] = cot_answer
            trans_data['smiles'] = smiles
            train_dataset.append(trans_data)

        if not os.path.exists(sft_path):
            os.mknod(sft_path)

        with open(sft_path, 'w', encoding="utf-8") as part_f:
            json.dump(train_dataset, part_f, indent=4) 
        print(f'Saved {len(train_dataset)} samples in {sft_path}')

async def main():
    # await process_json(
    #     json_path="/fs_mol/guojianz/projects/grpo/desc2mol/origin_data/desc2mol_train.json",
    #     out_path="/fs_mol/guojianz/projects/grpo/desc2mol/sft_data/train_for_gpt4_prompt.jsonl",
    #     sft_path="/fs_mol/guojianz/projects/grpo/desc2mol/sft_data/hme_desc2mol_processed_train.json",
    #     split="train"
    # )
    await process_json(
        json_path="/fs_mol/guojianz/projects/grpo/desc2mol/origin_data/desc2mol_test.json",
        out_path="/fs_mol/guojianz/projects/grpo/desc2mol/sft_data/test_for_gpt4_prompt.jsonl",
        sft_path="/fs_mol/guojianz/projects/grpo/desc2mol/sft_data/hme_desc2mol_processed_test.json",
        split="test"
    )


if __name__ == "__main__":
    asyncio.run(main())