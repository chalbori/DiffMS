"""
Preprocessing script: info_spectrum_sop.csv → DiffMS-compatible data/sop/

Selection logic per (c_id, prec_type):
  - Only [M+H]+ and [M+Na]+ adducts
  - Prefer qualified=1; if none, use row with highest TIC
  - Filter out molecules with atoms outside {C, O, P, N, S, Cl, F, H}

Outputs:
  data/sop/spec_files/{spec_id}.ms
  data/sop/subformulae/default_subformulae/{spec_id}.json
  data/sop/labels.tsv
  data/sop/split.tsv
  data/sop/n_counts.txt, atom_types.txt, edge_types.txt, valencies.txt  (copied from MSG)
  configs/dataset/sop.yaml
"""
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from src.mist.utils.spectra_utils import assign_subforms, max_inten_spec

RDLogger.DisableLog("rdApp.*")

CSV_PATH    = PROJECT_DIR / "info_spectrum_sop.csv"
OUT_DIR     = PROJECT_DIR / "data" / "sop"
MSG_DIR     = PROJECT_DIR / "data" / "msg"
CONFIG_DIR  = PROJECT_DIR / "configs" / "dataset"

SUPPORTED_IONS = {"[M+H]+", "[M+Na]+"}
ALLOWED_ATOMS  = {"C", "O", "P", "N", "S", "Cl", "F", "H"}


# ── CSV loading ──────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    """Read CSV that has numpy-binary blobs in the peaks column."""
    with open(path, "rb") as f:
        raw = f.read()
    cleaned = raw.replace(b"\x00", b"__NUL__")
    return pd.read_csv(io.BytesIO(cleaned), encoding="latin-1")


def parse_peaks(peaks_field: str) -> np.ndarray:
    """Restore peaks field (str with __NUL__ placeholders) to (N,2) array."""
    restored = peaks_field.replace("__NUL__", "\x00").encode("latin-1")
    return np.load(io.BytesIO(restored))


# ── Spectrum processing ──────────────────────────────────────────────────────

def process_peaks(peaks: np.ndarray, parentmass: float,
                  max_peaks: int = 60, inten_thresh: float = 0.001) -> np.ndarray:
    mz, inten = peaks[:, 0], peaks[:, 1]
    mask = mz <= parentmass + 1
    mz, inten = mz[mask], inten[mask]
    if len(inten) == 0:
        return np.zeros((0, 2))
    inten = np.sqrt(inten / inten.max())   # sqrt + normalize (MIST convention)
    spec = np.stack([mz, inten], axis=1)
    return max_inten_spec(spec, max_num_inten=max_peaks, inten_thresh=inten_thresh)


# ── Molecule utilities ───────────────────────────────────────────────────────

def smiles_to_formula(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcMolFormula(mol) if mol else None


def has_only_allowed_atoms(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return all(a.GetSymbol() in ALLOWED_ATOMS for a in mol.GetAtoms())


# ── File writers ─────────────────────────────────────────────────────────────

def write_ms_file(path: Path, spec_id: str, formula: str, parentmass: float,
                  ion_type: str, smiles: str, inchikey: str, spec: np.ndarray):
    lines = [
        f">compound {spec_id}",
        f">formula {formula}",
        f">parentmass {parentmass}",
        f">ionization {ion_type}",
        ">InChi None",
        f">InChiKey {inchikey}",
        f"#smiles {smiles}",
        "",
        ">ms2peaks",
    ]
    for mz_val, inten_val in spec:
        lines.append(f"{mz_val} {inten_val}")
    path.write_text("\n".join(lines))


def write_sop_config():
    content = f"""\
name: msg
remove_h: null
stats_dir: null
datadir: '../../../data/sop'
filter: False
denoise_nodes: False
merge: 'downproject_4096'
morgan_nbits: 2048
morgan_r: 2
split_file: '../../../data/sop/split.tsv'
spec_features: 'peakformula'
mol_features: 'fingerprint'
subform_folder: '../../../data/sop/subformulae/default_subformulae'
augment_data: False
remove_prob: 0.1
remove_weights: 'exp'
inten_prob: 0.1
inten_transform: 'float'
cls_type: 'ms1'
magma_aux_loss: False
labels_file: '../../../data/sop/labels.tsv'
spec_folder: '../../../data/sop/spec_files'
cache_featurizers: True
set_pooling: 'cls'
max_count: null
"""
    (CONFIG_DIR / "sop.yaml").write_text(content)


# ── Selection logic ──────────────────────────────────────────────────────────

def select_best(group: pd.DataFrame) -> pd.Series:
    """Per (c_id, prec_type): qualified=1 rows first, then highest TIC."""
    qualified = group[group["qualified"] == 1]
    pool = qualified if len(qualified) > 0 else group
    return pool.loc[pool["tic"].idxmax()]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading CSV...")
    df = load_csv(CSV_PATH)
    print(f"  Total rows       : {len(df):,}")

    # Step 1: adduct filter
    df = df[df["prec_type"].isin(SUPPORTED_IONS)].copy()
    print(f"  [M+H]+/[M+Na]+  : {len(df):,}")

    # Step 2: best per (c_id, prec_type)
    selected = (
        df.groupby(["c_id", "prec_type"], group_keys=False)
          .apply(select_best)
          .reset_index(drop=True)
    )
    print(f"  After dedup      : {len(selected):,}  "
          f"({selected['c_id'].nunique()} unique compounds)")

    # Step 3: atom filter
    atom_ok = selected["smiles"].apply(has_only_allowed_atoms)
    n_filtered = (~atom_ok).sum()
    selected = selected[atom_ok].reset_index(drop=True)
    print(f"  After atom filter: {len(selected):,}  (removed {n_filtered} with unsupported atoms)")

    # Create output directories
    spec_dir    = OUT_DIR / "spec_files"
    subform_dir = OUT_DIR / "subformulae" / "default_subformulae"
    spec_dir.mkdir(parents=True, exist_ok=True)
    subform_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: generate files
    print(f"\nGenerating .ms + subformulae for {len(selected)} spectra...")
    labels_rows = []
    failed = []

    for i, (_, row) in enumerate(selected.iterrows(), 1):
        spec_id   = f"SOPID{row['spec_id']}"
        smiles    = str(row["smiles"])
        inchikey  = str(row["inchikey"])
        ion_type  = str(row["prec_type"])
        parentmass = float(row["prec_mz"])

        formula = smiles_to_formula(smiles)
        if formula is None:
            failed.append(spec_id)
            continue

        try:
            raw_peaks = parse_peaks(row["peaks"])
            spec = process_peaks(raw_peaks, parentmass)
        except Exception:
            failed.append(spec_id)
            continue

        if len(spec) == 0:
            failed.append(spec_id)
            continue

        write_ms_file(spec_dir / f"{spec_id}.ms",
                      spec_id, formula, parentmass, ion_type, smiles, inchikey, spec)

        subform = assign_subforms(formula, spec, ion_type)
        with open(subform_dir / f"{spec_id}.json", "w") as fp:
            json.dump(subform, fp)

        labels_rows.append({
            "dataset": "sop",
            "spec": spec_id,
            "ionization": ion_type,
            "formula": formula,
            "smiles": smiles,
            "inchikey": inchikey,
            "instrument": str(row.get("instrument_type", "")),
            "c_id": str(row["c_id"]),
        })

        if i % 100 == 0:
            print(f"  {i}/{len(selected)}...")

    print(f"  Done. Success: {len(labels_rows)}, Failed/skipped: {len(failed)}")

    # labels.tsv (c_id excluded from TSV — internal use only)
    labels_df = pd.DataFrame(labels_rows)
    labels_df.drop(columns=["c_id"]).to_csv(OUT_DIR / "labels.tsv", sep="\t", index=False)

    # split.tsv: first 2 → train/val (needed by dataset infos), rest → test
    ids = labels_df["spec"].tolist()
    with open(OUT_DIR / "split.tsv", "w") as fp:
        fp.write("name\tsplit\n")
        fp.write(f"{ids[0]}\ttrain\n")
        if len(ids) > 1:
            fp.write(f"{ids[1]}\tval\n")
        for sid in ids[2:]:
            fp.write(f"{sid}\ttest\n")

    # Copy MSG stats files (model was trained on MSG; reuse its distributions)
    for fname in ["n_counts.txt", "atom_types.txt", "edge_types.txt", "valencies.txt"]:
        src = MSG_DIR / fname
        if src.exists():
            shutil.copy(src, OUT_DIR / fname)
        else:
            print(f"  WARNING: {src} not found — stats file not copied")

    # Dataset config
    write_sop_config()

    # Final summary
    W = 52
    print(f"\n{'=' * W}")
    print(f"  Output dir   : {OUT_DIR}")
    print(f"  Total ready  : {len(labels_rows):>5}")
    mh  = (labels_df['ionization'] == '[M+H]+').sum()
    mna = (labels_df['ionization'] == '[M+Na]+').sum()
    print(f"    [M+H]+     : {mh:>5}")
    print(f"    [M+Na]+    : {mna:>5}")
    print(f"  Test spectra : {max(0, len(labels_rows) - 2):>5}")
    print(f"  Compounds    : {labels_df['c_id'].nunique():>5}")
    print(f"  Config       : configs/dataset/sop.yaml")
    print(f"{'=' * W}")

    if failed:
        print(f"\nFailed ({len(failed)}): {failed[:5]}{'...' if len(failed)>5 else ''}")


if __name__ == "__main__":
    main()
