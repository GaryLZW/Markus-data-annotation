import torch
from molscribe import MolScribe
from huggingface_hub import hf_hub_download
from rdkit import Chem
import cv2
# import pytesseract
import pandas as pd
import os
from rdkit import Chem
import re


def ocsr_predict(img_path, model_name='swin_base_char_aux_1m.pth'):
    # 初始化 MolScribe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = hf_hub_download('yujieq/MolScribe', model_name)

    model = MolScribe(ckpt_path, device=torch.device(device))
    
    try:
        result = model.predict_image_file(img_path, return_atoms_bonds=True)
        smiles = result["smiles"]
        return smiles
    except Exception as e:
        return None


def needs_review(img_path, smiles):

    ABSTRACT_KEYWORDS = ["tBu", "Bn", "Ph", "R", "Ar", "PG"]
    
    # RDKit 合法性
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True, "Invalid SMILES"
    # OCR 检测抽象基团
    # img = cv2.imread(img_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # text = pytesseract.image_to_string(gray)
    # for kw in ABSTRACT_KEYWORDS:
    #     if kw in text and kw not in smiles:
    #         return True, f"Detected {kw} but not in SMILES"
    return False, ""

def extract_and_judge_smiles(processed_image_path, raw_image_path):
    # results = []

    # for fname in os.listdir(image_dir):
        # 
        # if not fname.endswith(".png"):
            # continue

    
    smiles = ocsr_predict(processed_image_path)
    
        
    if smiles is None:
        smiles = ocsr_predict(raw_image_path)
    
    if smiles is None:
        review, reason = True, "OCSR failed"
    else:
        review, reason = needs_review(raw_image_path, smiles)
    
    return {"image": raw_image_path[-12:],
    "smiles": smiles,
    "need_review": review,
    "reason": reason
    }
    
    # results.append({
        # "image": fname,
        # "smiles": smiles,
        # "need_review": review,
        # "reason": reason
    # })

    # df = pd.DataFrame(results)

    # df.to_csv(output_csv, index=False)


def make_rdkit_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True, rootedAtAtom=0)

ABSTRACT_GROUPS = {"R", "X", "Y", "Z", "Ph", "Bn", "Ar"}
def build_extension(atom_map, ring_map, circle_map):
    parts = []
    for idx, name in atom_map.items():
        parts.append(f"<a>{idx}:{name}</a>")
    for r_idx, name in ring_map.items():
        parts.append(f"<r>{r_idx}:{name}</r>")
    for c_idx, name in circle_map.items():
        parts.append(f"<c>{c_idx}:{name}</c>")
    return "".join(parts)

SEP = "<sep>"
def to_e_smiles(rdkit_smiles, extension_xml):
    if extension_xml:
        return rdkit_smiles + SEP + extension_xml
    else:
        return rdkit_smiles


INDEX_GROUP_PATTERN = re.compile(r"^\d+:[A-Za-z0-9\[\]_]+$")

TAG_MAP = {
    "a": "atom",
    "r": "ring",
    "c": "circle"
}

def validate_extension(xml: str):
    """
    返回:
      valid: bool
      error: str or None
    """
    if not xml.strip():
        return True, None

    tokens = re.findall(r"<(/?)(\w+)>(.*?)</\2>", xml)
    if not tokens:
        return False, "No valid XML tags found"

    seen = set()

    for _, tag, content in tokens:
        if tag not in TAG_MAP:
            return False, f"Invalid tag <{tag}>"

        if not INDEX_GROUP_PATTERN.match(content):
            return False, f"Invalid content in <{tag}>: {content}"

        key = (tag, content)
        if key in seen:
            return False, f"Duplicate tag: <{tag}>{content}</{tag}>"
        seen.add(key)

    return True, None
