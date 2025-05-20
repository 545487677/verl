# Copyright 2025 Guojiang Zhao
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
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from Levenshtein import distance as lev
from rdkit import rdBase
from rdkit import RDLogger
from rdkit import Chem
rdBase.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')


def extract_solution(solution_str: str) -> str:
    """
    Extract the last <answer>...</answer> block from the solution string.
    """
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None


def is_valid_smiles(smi: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol is not None
    except:
        return False


def exact_string_match(pred_smi: str, gt_smi: str) -> float:
    try:
        can_pred = Chem.MolToSmiles(Chem.MolFromSmiles(pred_smi), canonical=True)
        can_gt = Chem.MolToSmiles(Chem.MolFromSmiles(gt_smi), canonical=True)
        return 1.0 if can_pred == can_gt else 0.0
    except:
        return 0.0


def exact_structure_match(pred_smi: str, gt_smi: str) -> float:
    try:
        m1, m2 = Chem.MolFromSmiles(pred_smi), Chem.MolFromSmiles(gt_smi)
        return 1.0 if Chem.MolToInchi(m1) == Chem.MolToInchi(m2) else 0.0
    except:
        return 0.0


def property_similarity(pred_smi: str, gt_smi: str) -> float:
    try:
        mol1, mol2 = Chem.MolFromSmiles(pred_smi), Chem.MolFromSmiles(gt_smi)
        props1 = np.array([Descriptors.MolWt(mol1), Descriptors.MolLogP(mol1), Descriptors.TPSA(mol1)])
        props2 = np.array([Descriptors.MolWt(mol2), Descriptors.MolLogP(mol2), Descriptors.TPSA(mol2)])
        diff = np.abs(props1 - props2)
        return float(np.exp(-np.mean(diff) / 10))
    except:
        return 0.0


def fingerprint_similarity_scores(pred_smi: str, gt_smi: str):
    try:
        mol1, mol2 = Chem.MolFromSmiles(pred_smi), Chem.MolFromSmiles(gt_smi)
        maccs_sim = DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(mol1), MACCSkeys.GenMACCSKeys(mol2))
        rdk_sim = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2))
        morgan_sim = DataStructs.TanimotoSimilarity(
            AllChem.GetMorganFingerprint(mol1, 2),
            AllChem.GetMorganFingerprint(mol2, 2)
        )
        return maccs_sim, rdk_sim, morgan_sim
    except:
        return 0.0, 0.0, 0.0


def smiles_levenshtein(pred_smi: str, gt_smi: str, normalize_len: int = 100) -> float:
    try:
        return 1.0 - lev(pred_smi, gt_smi) / normalize_len
    except:
        return 0.0


# def compute_score(solution_str: str, ground_truth: str) -> float:
    
#     pred_smi = extract_solution(solution_str)
#     if pred_smi is None:
#         return 0.0  

#     if not is_valid_smiles(pred_smi):
#         return 0.0  
    
#     exact_text = exact_string_match(pred_smi, ground_truth)
#     exact_struct = exact_structure_match(pred_smi, ground_truth)
#     prop_sim = property_similarity(pred_smi, ground_truth)
#     maccs_sim, rdk_sim, morgan_sim = fingerprint_similarity_scores(pred_smi, ground_truth)
#     lev_sim = smiles_levenshtein(pred_smi, ground_truth)

#     weights = {
#         "exact_text_match": 0.85, # 
#         "exact_struct_match": 0.1,
#         # "morgan_similarity": 0.15,
#         # "property_similarity": 0.1,
#         # "smiles_levenshtein": 0.05,
#         # "rdk_similarity": 0.05,
#         # "maccs_similarity": 0.05,
#         "format_text": 0.05,
#     }

#     score = (
#         weights["exact_text_match"] * exact_text +
#         weights["exact_struct_match"] * exact_struct +
#         # weights["property_similarity"] * prop_sim +
#         # weights["morgan_similarity"] * morgan_sim +
#         # weights["rdk_similarity"] * rdk_sim +
#         # weights["maccs_similarity"] * maccs_sim +
#         # weights["smiles_levenshtein"] * lev_sim + 
#         weights["format_text"] * 1
#     )
    
#     return float(score)

def compute_score(solution_str: str, ground_truth: str) -> float:
    pred_smi = extract_solution(solution_str)
    if pred_smi is None or not is_valid_smiles(pred_smi):
        return 0.0

    exact_text = exact_string_match(pred_smi, ground_truth)
    exact_struct = exact_structure_match(pred_smi, ground_truth)
    _, _, morgan_sim = fingerprint_similarity_scores(pred_smi, ground_truth)

    if exact_text == 1.0:
        return 1.0
    elif exact_struct == 1.0:
        return 0.9
    else:
        return 0.3 + 0.6 * morgan_sim  