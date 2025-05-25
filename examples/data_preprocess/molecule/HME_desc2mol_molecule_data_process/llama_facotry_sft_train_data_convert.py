import os
import json
import re

def extract_solution(solution_str: str) -> str:
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        return matches[-1].group(1).strip()
    return None


all_f_path = "/vepfs/fs_projects/FunMG/LLM/dataset/mol_grpo/desc2mol_sft_cot_parquet/train_for_gpt4_prompt_max.jsonl"
part_f_path = "/fs_mol/guojianz/projects/grpo/desc2mol/sft_data/hme_desc2mol_processed_train.json"
train_dataset_1k = []
total = 0
filtered_dot = 0
filtered_unicode = 0
filtered_non_ascii = 0
filtered_answer_mismatch = 0
accepted = 0
with open(all_f_path, 'r') as all_f:
    for line in all_f.readlines():
        total += 1
        data = json.loads(line)
        trans_data = {}
        trans_data["instruction"] = "You are a professional biochemist designing molecular structures. Please generate the molecular structure (SMILES) based on the following description.\n\n Please think step by step and return the final answer in <answer> </answer> tags, for example <answer> CCCCC[C@@H]1O[C@@H]1/C=C/C(O)C/C=C\\C/C=C\\CCCC(=O)[O-] </answer>. Do not use <|...|> or other formats. The answer **must** be returned in the format: <answer> CCCCC[C@@H]1O[C@@H]1/C=C/C(O)C/C=C\\C/C=C\\CCCC(=O)[O-] </answer>."
        trans_data["input"] = data["description"]
        smiles = data['smiles']
        if '.' in smiles:
            filtered_dot += 1
            continue

        cot_answer = data["cot_answer"]
        if re.search(r"\\u[0-9a-fA-F]{4}", cot_answer):
            filtered_unicode += 1
            continue
        if any(ord(c) > 127 for c in cot_answer):
            filtered_non_ascii += 1
            continue
        
        pred = extract_solution(cot_answer)
        if pred != smiles:
            print('invalid')
            filtered_answer_mismatch += 1
            continue
        trans_data["output"] = cot_answer
        trans_data['smiles'] = smiles
        train_dataset_1k.append(trans_data)
        accepted += 1
    if not os.path.exists(part_f_path):
        os.mknod(part_f_path)
    with open(part_f_path, 'w', encoding="utf-8") as part_f:
        json.dump(train_dataset_1k, part_f, indent=4) #保存的json文件按照4格缩进

print(f"📊 总样本数: {total}")
print(f"✅ 保留样本数: {accepted}")
print(f"❌ 过滤（包含 '.' 的 SMILES）: {filtered_dot}")
print(f"❌ 过滤（包含 unicode 转义字符）: {filtered_unicode}")
print(f"❌ 过滤（非 ASCII 字符）: {filtered_non_ascii}")
print(f"❌ 过滤（<answer> 提取不等于 smiles）: {filtered_answer_mismatch}")