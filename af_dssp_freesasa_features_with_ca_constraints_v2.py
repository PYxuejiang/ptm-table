#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AF residue-level structure features extraction for PTM 3D analysis mainline (FIXED RSA).

Key fix (per your decision):
  - DSSP returned value (0-1) is treated as relative solvent accessibility (dssp_rsa).
  - Do NOT compute RSA by dividing by MAX_ASA.
  - Absolute exposure uses FreeSASA sasa_total/main/side.

Input:
  - PTM site table (pos+neg): ptm_all_pos_neg.tsv
  - AF structure files: .pdb/.cif/.bcif (+ gz)

Output:
  1) Residue-level background table:
      af_uniprot_residue_features_parallel.tsv
  2) PTM site-level merged table:
      ptm_af_dssp_sasa_parallel.tsv

Features:
  - DSSP: dssp_ss, dssp_rsa, SS_class
  - FreeSASA: sasa_total/main/side, side_frac
  - CA coords: ca_x, ca_y, ca_z (+ has_ca)
  - pLDDT: from B-factor (AF PDB convention)
  - RSA_bin, exposed_flag, buried_flag
  - norm_sasa_total (within protein) + ConstraintScore:
      ConstraintScore = (1 - dssp_rsa) + (1 - norm_sasa_total)

要求的口径是：
DSSP 返回的 0–1 直接作为 dssp_rsa（也可理解 rel RSA）

不再用 MAX_ASA

绝对暴露度用 FreeSASA sasa_total（明确要这样做）
"""

from __future__ import annotations

import re
import gzip
import tempfile
import multiprocessing
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from Bio.PDB import PDBParser, MMCIFParser, DSSP, is_aa, PDBIO
import freesasa


# =========================
# 1) Path config
# =========================
PTM_ALL_TSV = "/data6/py/neg_site_v2/ptm_all_pos_neg.tsv"

AF_STRUCT_DIR = "/data1/py/PDB_AF/PDB_PTM结合蛋白质三维结构/PDB_PTM结合蛋白质三维结构"

OUT_DIR = Path("/data6/py/DSSP+FreeSASA_AF_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RESIDUE_TSV = str(OUT_DIR / "af_uniprot_residue_features_parallel.tsv")
OUT_PTM_TSV = str(OUT_DIR / "ptm_af_dssp_sasa_parallel.tsv")

DSSP_EXE = "mkdssp"
N_WORKERS = min(32, multiprocessing.cpu_count() or 1)

RSA_EXPOSED_TH = 0.25
RSA_BURIED_TH = 0.05


def rsa_bin(rsa: float) -> str:
    if np.isnan(rsa):
        return "NA"
    if rsa < 0.20:
        return "buried"
    if rsa < 0.50:
        return "intermediate"
    return "exposed"


def ss_class(dssp_ss: str) -> str:
    s = (dssp_ss or "").strip()
    if not s or s == "?":
        return "NA"
    if s in ("H", "G", "I"):
        return "H"
    if s in ("E", "B"):
        return "E"
    return "C"


# =========================
# 2) Structure resolving / conversion
# =========================
def find_structure_file(struct_dir: Path, tag: str) -> Optional[Path]:
    exts = [".pdb", ".pdb.gz", ".cif", ".cif.gz", ".bcif", ".bcif.gz"]
    for ext in exts:
        p = struct_dir / f"{tag}{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def load_structure_any(path: Path, structure_id: str):
    suffix = path.suffix.lower()
    is_gz = (suffix == ".gz")
    real_path = path

    if is_gz:
        inner = Path(path.stem).suffix.lower()
        with gzip.open(path, "rb") as f:
            data = f.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=inner if inner else ".pdb")
        tmp.write(data)
        tmp.close()
        real_path = Path(tmp.name)

    ext = real_path.suffix.lower()
    if ext == ".pdb":
        parser = PDBParser(QUIET=True)
        st = parser.get_structure(structure_id, str(real_path))
    else:
        parser = MMCIFParser(QUIET=True)
        st = parser.get_structure(structure_id, str(real_path))

    return st, real_path, is_gz


def write_temp_pdb(structure, structure_id: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    tmp.close()
    io = PDBIO()
    io.set_structure(structure)
    io.save(tmp.name)
    return Path(tmp.name)


# =========================
# 3) DSSP / FreeSASA per-residue extraction
# =========================
def run_dssp_on_pdb(pdb_path: Path, structure_id: str):
    """
    Return dict:
      key = (chain_id, resseq, icode) -> {dssp_aa, dssp_ss, dssp_rsa}

    NOTE:
      In your environment, Biopython DSSP d[3] is 0-1 scale.
      We treat it as dssp_rsa directly.
    """
    parser = PDBParser(QUIET=True)
    st = parser.get_structure(structure_id, str(pdb_path))
    model = next(st.get_models())
    dssp = DSSP(model, str(pdb_path), dssp=DSSP_EXE)

    out: Dict[Tuple[str, int, str], dict] = {}
    for key in dssp.keys():
        chain_id, res_id = key
        resseq, icode = res_id[1], (res_id[2].strip() if isinstance(res_id[2], str) else str(res_id[2]).strip())
        icode = icode or ""
        d = dssp[key]
        aa = d[1]
        ss = d[2]
        rsa_rel = float(d[3])  # 0-1 in your current outputs

        # safety clip
        if np.isnan(rsa_rel):
            rsa_rel = np.nan
        else:
            rsa_rel = max(0.0, min(1.0, rsa_rel))

        out[(chain_id, int(resseq), icode)] = {"dssp_aa": aa, "dssp_ss": ss, "dssp_rsa": rsa_rel}
    return out


def run_freesasa_on_pdb(pdb_path: Path) -> Dict[Tuple[str, int, str], dict]:
    structure = freesasa.Structure(str(pdb_path))
    result = freesasa.calc(structure)
    res_areas_by_chain = result.residueAreas()
    out: Dict[Tuple[str, int, str], dict] = {}
    if not res_areas_by_chain:
        return out

    for chain_id, res_dict in res_areas_by_chain.items():
        chain_id = str(chain_id)
        for res_key, area in res_dict.items():
            resseq = None
            icode = ""

            if hasattr(res_key, "residueNumber"):
                try:
                    resseq = int(res_key.residueNumber)
                except Exception:
                    resseq = None
                if hasattr(res_key, "insertionCode"):
                    icode = str(res_key.insertionCode or "")

            if resseq is None:
                s = str(res_key)
                m = re.search(r"(-?\d+)", s)
                if m:
                    resseq = int(m.group(1))
                else:
                    continue

            out[(chain_id, int(resseq), icode)] = {
                "sasa_total": float(area.total),
                "sasa_main": float(area.mainChain),
                "sasa_side": float(area.sideChain),
            }
    return out


def build_uniprot_mapping_for_chain(chain) -> Dict[Tuple[str, int, str], int]:
    mapping = {}
    pos = 0
    for res in chain:
        if not is_aa(res, standard=True):
            continue
        pos += 1
        chain_id = chain.id
        resseq = int(res.id[1])
        icode = res.id[2].strip() if isinstance(res.id[2], str) else str(res.id[2]).strip()
        icode = icode or ""
        mapping[(chain_id, resseq, icode)] = pos
    return mapping


def extract_ca_and_plddt(res) -> Tuple[float, float, float, int, float]:
    if "CA" in res:
        ca = res["CA"]
        x, y, z = map(float, ca.coord)
        b = float(getattr(ca, "bfactor", np.nan))
        return x, y, z, 1, b
    return np.nan, np.nan, np.nan, 0, np.nan


def extract_residue_features_for_protein(struct_path: Path, protein_id: str, specie: str) -> List[dict]:
    st, real_path, was_gz = load_structure_any(struct_path, protein_id)
    model = next(st.get_models())
    chain_ids = [ch.id for ch in model.get_chains()]
    chain = model["A"] if "A" in chain_ids else model[chain_ids[0]]

    tmp_pdb = write_temp_pdb(st, protein_id)

    try:
        dssp_dict = run_dssp_on_pdb(tmp_pdb, protein_id)
    except Exception:
        dssp_dict = {}
    try:
        sasa_dict = run_freesasa_on_pdb(tmp_pdb)
    except Exception:
        sasa_dict = {}

    uni_map = build_uniprot_mapping_for_chain(chain)

    rows: List[dict] = []
    for res in chain:
        if not is_aa(res, standard=True):
            continue
        chain_id = chain.id
        resseq = int(res.id[1])
        icode = res.id[2].strip() if isinstance(res.id[2], str) else str(res.id[2]).strip()
        icode = icode or ""
        key = (chain_id, resseq, icode)
        if key not in uni_map:
            continue

        pos = int(uni_map[key])

        # DSSP aa fallback
        aa1 = "X"
        if key in dssp_dict:
            aa1 = (dssp_dict[key].get("dssp_aa") or "X").strip()

        ca_x, ca_y, ca_z, has_ca, plddt = extract_ca_and_plddt(res)

        feat = {
            "protein_id": protein_id,
            "specie": specie,
            "chain_id": chain_id,
            "position": pos,
            "AA": aa1,
            "resseq": resseq,
            "icode": icode,
            "ca_x": ca_x,
            "ca_y": ca_y,
            "ca_z": ca_z,
            "has_ca": has_ca,
            "plddt": plddt,
        }

        # DSSP: SS + RSA(rel)
        if key in dssp_dict:
            feat["dssp_ss"] = dssp_dict[key]["dssp_ss"]
            feat["dssp_rsa"] = float(dssp_dict[key]["dssp_rsa"])
        else:
            feat["dssp_ss"] = "?"
            feat["dssp_rsa"] = np.nan

        feat["SS_class"] = ss_class(feat["dssp_ss"])

        # FreeSASA: absolute exposure
        if key in sasa_dict:
            feat["sasa_total"] = sasa_dict[key]["sasa_total"]
            feat["sasa_main"] = sasa_dict[key]["sasa_main"]
            feat["sasa_side"] = sasa_dict[key]["sasa_side"]
        else:
            feat["sasa_total"] = np.nan
            feat["sasa_main"] = np.nan
            feat["sasa_side"] = np.nan

        if np.isnan(feat["sasa_total"]) or feat["sasa_total"] <= 0 or np.isnan(feat["sasa_side"]):
            feat["side_frac"] = np.nan
        else:
            feat["side_frac"] = float(feat["sasa_side"] / feat["sasa_total"])

        # flags/bins/niche based on dssp_rsa
        rsa_val = feat["dssp_rsa"]
        feat["exposed_flag"] = int((not np.isnan(rsa_val)) and (rsa_val >= RSA_EXPOSED_TH))
        feat["buried_flag"] = int((not np.isnan(rsa_val)) and (rsa_val <= RSA_BURIED_TH))

        rb = rsa_bin(rsa_val)
        feat["RSA_bin"] = rb
        feat["niche_key"] = f"{feat['SS_class']}|{rb}" if feat["SS_class"] != "NA" and rb != "NA" else "NA"

        rows.append(feat)

    # cleanup temp files
    try:
        tmp_pdb.unlink(missing_ok=True)
    except Exception:
        pass
    if was_gz:
        try:
            Path(real_path).unlink(missing_ok=True)
        except Exception:
            pass

    return rows


def worker_one(args: Tuple[str, str, str]) -> List[dict]:
    specie, uid, struct_dir = args
    struct_dir = Path(struct_dir)
    p = find_structure_file(struct_dir, uid)
    if not p:
        return []
    try:
        return extract_residue_features_for_protein(p, uid, specie)
    except Exception:
        return []


def add_constraint_score(residue_df: pd.DataFrame) -> pd.DataFrame:
    """
    norm_sasa_total: min-max normalize within protein (using FreeSASA sasa_total)
    ConstraintScore = (1 - dssp_rsa) + (1 - norm_sasa_total)
    """
    df = residue_df.copy()

    def _norm(group: pd.DataFrame) -> pd.Series:
        x = pd.to_numeric(group["sasa_total"], errors="coerce").astype(float)
        v = x.values
        if not np.any(~np.isnan(v)):
            return pd.Series([np.nan] * len(group), index=group.index)
        xmin = np.nanmin(v)
        xmax = np.nanmax(v)
        if np.isnan(xmin) or np.isnan(xmax) or xmax == xmin:
            return pd.Series([np.nan] * len(group), index=group.index)
        return (x - xmin) / (xmax - xmin)

    df["norm_sasa_total"] = df.groupby("protein_id", group_keys=False).apply(_norm)
    df["ConstraintScore"] = (1.0 - pd.to_numeric(df["dssp_rsa"], errors="coerce").astype(float)) + \
                            (1.0 - pd.to_numeric(df["norm_sasa_total"], errors="coerce").astype(float))
    return df


def main():
    struct_dir = Path(AF_STRUCT_DIR)
    if not struct_dir.exists():
        raise FileNotFoundError(f"AF_STRUCT_DIR not found: {struct_dir}")

    ptm = pd.read_csv(PTM_ALL_TSV, sep="\t", low_memory=False)
    ptm["Protein ID"] = ptm["Protein ID"].astype(str).str.strip()
    ptm["Specie"] = ptm["Specie"].astype(str).str.upper().str.strip()
    ptm["Site"] = ptm["Site"].astype(int)

    proteins = ptm[["Specie", "Protein ID"]].drop_duplicates().values.tolist()

    proteins_with_struct = []
    for sp, uid in proteins:
        if find_structure_file(struct_dir, uid):
            proteins_with_struct.append((sp, uid, str(struct_dir)))

    print(f"[INFO] PTM proteins: {len(proteins)} | with AF structure: {len(proteins_with_struct)}")
    if not proteins_with_struct:
        raise RuntimeError("No AF structure files found for proteins in PTM table.")

    all_rows: List[dict] = []
    print(f"[INFO] Start parallel AF(DSSP+FreeSASA+CA) with {N_WORKERS} workers")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for rows in tqdm(
            ex.map(worker_one, proteins_with_struct),
            total=len(proteins_with_struct),
            desc="AF residue features per protein",
        ):
            if rows:
                all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No residue features generated. Check DSSP/FreeSASA or file formats.")

    residue_df = pd.DataFrame(all_rows)
    residue_df.sort_values(["protein_id", "position"], inplace=True)

    residue_df = add_constraint_score(residue_df)

    Path(OUT_RESIDUE_TSV).parent.mkdir(parents=True, exist_ok=True)
    residue_df.to_csv(OUT_RESIDUE_TSV, sep="\t", index=False)
    print(f"[OK] Saved residue-level AF features: {OUT_RESIDUE_TSV} | shape={residue_df.shape}")

    ptm_ren = ptm.rename(columns={"Protein ID": "protein_id", "Site": "position"})
    merged = residue_df.merge(ptm_ren, how="inner", on=["protein_id", "position"])
    merged.to_csv(OUT_PTM_TSV, sep="\t", index=False)
    print(f"[OK] Saved PTM+AF merged table: {OUT_PTM_TSV} | shape={merged.shape}")


if __name__ == "__main__":
    main()

