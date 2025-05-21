import os
import json
import argparse
import pandas as pd
from rdkit import Chem
from transformers import AutoTokenizer
from verl.utils.hdfs_io import copy, makedirs  
import os
import json
import pandas as pd
from rdkit import Chem
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit import RDLogger
from typing import List, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
from rdkit.Chem import Draw
rdBase.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')


tail_instruction = (
    "Let's think step by step and return the final answer in <answer> </answer> tags, for example <answer> CCCCC[C@@H]1O[C@@H]1/C=C/C(O)C/C=C\\C/C=C\\CCCC(=O)[O-] </answer>. Do not use <|...|> or other formats. The answer **must** be returned in the format: <answer> CCCCC[C@@H]1O[C@@H]1/C=C/C(O)C/C=C\\C/C=C\\CCCC(=O)[O-] </answer>."
)

one_shot_description = (
    "The molecule is an epoxy(hydroxy)icosatrienoate that is the conjugate base of "
    "11 hydroxy-(14R,15S)-epoxy-(5Z,8Z,12E)-icosatrienoic acid, obtained by deprotonation of the carboxy group; "
    "major species at pH 7.3. It is a conjugate base of an 11 hydroxy-(14R,15S)-epoxy-(5Z,8Z,12E)-icosatrienoic acid."
)
one_shot_smiles = "CCCCC[C@@H]1O[C@@H]1/C=C/C(O)C/C=C\\C/C=C\\CCCC(=O)[O-]"

instruction_prefix = (
    "Please generate the molecular structure (SMILES) based on the following description.\n\n"
    "### Example:\n"
    f"Description: {one_shot_description}\n"
    f"Answer: <answer> {one_shot_smiles} </answer>\n\n"
    "### Now try this:\n"
)

def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None

def load_and_filter(json_path: str, tokenizer, seen_smiles: set, split: str):
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    records = []
    for idx, item in enumerate(raw_data):
        smiles = item.get("smiles", "")
        description = item.get("description", "")
        fragments = item.get("fragments", "")
        if not is_valid_smiles(smiles) or smiles in seen_smiles:
            continue
        if one_shot_smiles == smiles:
            print('remove one-shot example')
            continue
        tokens = tokenizer.tokenize(smiles)
        if any(tok in {"[UNK]", "<unk>"} for tok in tokens):
            continue

        seen_smiles.add(smiles)

        full_prompt = instruction_prefix + "Description: " + description.strip() + "\n" + tail_instruction

        records.append({
            "data_source": "HME/desc2mol",
            "prompt": [{"role": "user", "content": full_prompt}],
            "ability": "molecule",
            "reward_model": {"style": "rule", "ground_truth": smiles},
            "extra_info": {
                "split": split,
                "index": idx,
                "description": description,
                "fragments": fragments,
                "smiles": smiles
            }
        })
    return records

def main(train_json, test_json, local_dir, hdfs_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(
        "/vepfs/fs_projects/FunMG/LLM/model_weight/qwen/Qwen2___5-7B-Instruct",
        trust_remote_code=True
    )
    os.makedirs(local_dir, exist_ok=True)

    seen = set()
    train_records = load_and_filter(train_json, tokenizer, seen, split="train")
    test_records = load_and_filter(test_json, tokenizer, seen, split="test")

    pd.DataFrame(train_records).to_parquet(os.path.join(local_dir, "train.parquet"), index=False)
    pd.DataFrame(test_records).to_parquet(os.path.join(local_dir, "test.parquet"), index=False)

    print(f"\u2705 Train size: {len(train_records)} | Test size: {len(test_records)}")
    if hdfs_dir:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default="/vepfs/fs_projects/FunMG/LLM/dataset/datasets--GreatCaptainNemo--HME_dataset/snapshots/934c6a76f50e1f90eb83abdbc8b5366dde00639e/desc2mol_train.json")
    parser.add_argument("--test_json", type=str, default="/vepfs/fs_projects/FunMG/LLM/dataset/datasets--GreatCaptainNemo--HME_dataset/snapshots/934c6a76f50e1f90eb83abdbc8b5366dde00639e/desc2mol_test.json")
    parser.add_argument("--local_dir", type=str, default="/vepfs/fs_projects/FunMG/LLM/dataset/mol_grpo/desc2mol_grpo_parquet")
    parser.add_argument("--hdfs_dir", type=str, default=None)
    args = parser.parse_args()

    main(args.train_json, args.test_json, args.local_dir, args.hdfs_dir)