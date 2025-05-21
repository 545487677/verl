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

def extract_solution(solution_str: str) -> str:
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        return matches[-1].group(1).strip()
    return None

api_model = APIModel()

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

    for item in tqdm(raw_data):
        if success_count >= 3000:
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
        out_path="/vepfs/fs_projects/FunMG/LLM/dataset/mol_grpo/desc2mol_sft_cot_parquet/train_for_gpt4_prompt.jsonl",
        split="train"
    )
    await process_json(
        json_path="/vepfs/fs_projects/FunMG/LLM/dataset/datasets--GreatCaptainNemo--HME_dataset/snapshots/934c6a76f50e1f90eb83abdbc8b5366dde00639e/desc2mol_test.json",
        out_path="/vepfs/fs_projects/FunMG/LLM/dataset/mol_grpo/desc2mol_sft_cot_parquet/test_for_gpt4_prompt.jsonl",
        split="test"
    )


if __name__ == "__main__":
    asyncio.run(main())