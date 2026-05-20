"""
Rerank existing predictions.csv by NLL score.

Loads a predictions.csv produced by infer_sop.py, re-scores each candidate
molecule using the model's ELBO (NLL), reorders pred_1..pred_k from most to
least likely, and saves a new CSV.  Recall metrics are recomputed so the two
files can be compared directly.

Usage:
    PYTHONPATH=src python scripts/rerank_by_nll.py \
        --input  outputs/sop_results/predictions.csv \
        --out    outputs/sop_results_reranked \
        --nll-mc 5
"""
import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Batch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from src.analysis.visualization import MolecularVisualization
from src.datasets import spec2mol_dataset
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src import utils as torch_utils

RDLogger.DisableLog("rdApp.*")

# ── Encoding constants (must match featurizers.py) ────────────────────────────
ATOM_DECODER = ['C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H']
ATOM_ENCODER = {a: i for i, a in enumerate(ATOM_DECODER)}
BOND_ENCODER = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
NUM_ATOM_TYPES = len(ATOM_DECODER)   # 8
NUM_EDGE_TYPES = len(BOND_ENCODER) + 1  # 5 (0=no bond)


# ── SMILES → dense graph tensors ─────────────────────────────────────────────

def smiles_to_dense(smiles: str):
    """Convert SMILES to (X, E) one-hot tensors.

    Returns (None, None) if SMILES is invalid, contains unknown atom types,
    or is a multi-component mixture (contains '.').
    X: (n, 8)  E: (n, n, 5)
    """
    if not smiles:
        return None, None
    # Reject multi-component SMILES (disconnected molecules)
    if "." in smiles:
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    n = mol.GetNumAtoms()
    type_idx = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ATOM_ENCODER:
            return None, None
        type_idx.append(ATOM_ENCODER[symbol])

    X = F.one_hot(torch.tensor(type_idx), num_classes=NUM_ATOM_TYPES).float()

    E = torch.zeros(n, n, NUM_EDGE_TYPES)
    E[:, :, 0] = 1.0  # default: no bond

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = BOND_ENCODER.get(bond.GetBondType())
        if bt is None:
            return None, None
        edge_idx = bt + 1
        E[i, j] = 0; E[i, j, edge_idx] = 1.0
        E[j, i] = 0; E[j, i, edge_idx] = 1.0

    return X, E  # (n, dx), (n, n, de)


def pad_graphs(graphs):
    """Pad a list of (X, E) pairs to the same max_n.

    Returns (X_batch, E_batch, node_mask): (bs, max_n, dx), (bs, max_n, max_n, de), (bs, max_n)
    """
    max_n = max(x.size(0) for x, _ in graphs)
    dx = NUM_ATOM_TYPES
    de = NUM_EDGE_TYPES
    bs = len(graphs)

    X_batch    = torch.zeros(bs, max_n, dx)
    E_batch    = torch.zeros(bs, max_n, max_n, de)
    node_mask  = torch.zeros(bs, max_n, dtype=torch.bool)

    for idx, (X, E) in enumerate(graphs):
        n = X.size(0)
        X_batch[idx, :n]       = X
        E_batch[idx, :n, :n]   = E
        node_mask[idx, :n]     = True
        # pad edges: set diagonal and out-of-mask to no-bond (class 0)
        E_batch[idx, n:, :, 0] = 1.0
        E_batch[idx, :, n:, 0] = 1.0

    return X_batch, E_batch, node_mask


# ── Metrics ───────────────────────────────────────────────────────────────────

def canonical_inchikey14(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    from rdkit.Chem.inchi import MolToInchiKey
    key = MolToInchiKey(mol)
    return key[:14] if key else None


def morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def recall_at_k(pred_smiles, true_key14: str, ks=(1, 5, 10, 20)) -> dict:
    result = {}
    for k in ks:
        result[f"recall@{k}"] = False
        for smi in pred_smiles[:k]:
            if smi and canonical_inchikey14(smi) == true_key14:
                result[f"recall@{k}"] = True
                break
    return result


def max_tanimoto(pred_smiles, true_fp) -> float:
    return max((tanimoto(morgan_fp(s), true_fp) for s in pred_smiles if s), default=0.0)


# ── Model utilities ───────────────────────────────────────────────────────────

def detect_device(preference: str) -> torch.device:
    if preference == "cuda":
        return torch.device("cuda")
    if preference == "mps":
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_spectrum(model, batch, device):
    output, aux = model.encoder(batch)
    data = batch["graph"]
    if model.merge == "mist_fp":
        y = aux["int_preds"][-1]
    elif model.merge in {"merge-encoder_output-linear", "merge-encoder_output-mlp"}:
        y = model.merge_function(aux["h0"])
    elif model.merge == "downproject_4096":
        y = model.merge_function(output)
    else:
        y = output
    return data, y


def load_model(cfg, ckpt_path, dataset_infos, train_metrics,
               visualization_tools, extra_features, domain_features):
    model = Spec2MolDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        visualization_tools=visualization_tools,
        extra_features=extra_features,
        domain_features=domain_features,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rerank infer_sop.py predictions by NLL")
    parser.add_argument("--input",      default="outputs/sop_results/predictions.csv",
                        help="predictions.csv from infer_sop.py")
    parser.add_argument("--out",        default="outputs/sop_results_reranked")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoints/diffms_msg.ckpt")
    parser.add_argument("--device",     choices=["auto","cpu","cuda","mps"], default="auto")
    parser.add_argument("--nll-mc",     type=int, default=5,
                        help="Monte Carlo samples for NLL (higher = more accurate, slower)")
    parser.add_argument("--max-test",   type=int, default=None)
    args = parser.parse_args()

    device    = detect_device(args.device)
    ckpt_path = (PROJECT_DIR / args.checkpoint).resolve()
    out_dir   = (PROJECT_DIR / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    in_csv    = PROJECT_DIR / args.input

    print(f"Device     : {device}")
    print(f"Checkpoint : {ckpt_path.name}")
    print(f"Input CSV  : {in_csv}")
    print(f"NLL MC     : {args.nll_mc}")

    # ── Config & dataset ──────────────────────────────────────────────────────
    print("\nLoading config & dataset...")
    sop_dir = PROJECT_DIR / "data" / "sop"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_DIR / "configs")):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=sop",
                "general.name=rerank_nll",
                "general.wandb=disabled",
                "general.gpus=0",
                "model.encoder_hidden_dim=512",
                "model.encoder_magma_modulo=2048",
                "train.eval_batch_size=1",
                "train.num_workers=0",
                f"dataset.datadir={sop_dir}",
                f"dataset.labels_file={sop_dir / 'labels.tsv'}",
                f"dataset.split_file={sop_dir / 'split.tsv'}",
                f"dataset.spec_folder={sop_dir / 'spec_files'}",
                f"dataset.subform_folder={sop_dir / 'subformulae' / 'default_subformulae'}",
            ],
        )

    datamodule    = spec2mol_dataset.Spec2MolDataModule(cfg)
    dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    extra_features  = (
        ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        if cfg.model.extra_features else DummyExtraFeatures()
    )
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )
    train_metrics       = TrainMolecularMetricsDiscrete(dataset_infos)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model...")
    model = load_model(cfg, ckpt_path, dataset_infos, train_metrics,
                       visualization_tools, extra_features, domain_features)
    model = model.to(device)

    # ── Load predictions ──────────────────────────────────────────────────────
    pred_df   = pd.read_csv(in_csv)
    labels_df = pd.read_csv(sop_dir / "labels.tsv", sep="\t")
    pred_cols = [c for c in pred_df.columns if c.startswith("pred_")]
    n_samples = len(pred_cols)

    test_loader = datamodule.test_dataloader()
    n_test = len(test_loader)
    if args.max_test:
        n_test = min(n_test, args.max_test)

    print(f"\nSpectra to rerank : {n_test}")
    print(f"Candidates/spectrum: {n_samples}")

    # ── Resume support ────────────────────────────────────────────────────────
    out_csv = out_dir / "predictions.csv"
    rows = []
    done_ids: set = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done_ids = set(prev["spec_id"].astype(str))
        print(f"Resuming: {len(done_ids)} spectra already done.\n")
    else:
        print(f"Starting...\n")

    t_total = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(x for i, x in enumerate(test_loader) if i < n_test):
            spec_id = str(batch["names"][0])

            if spec_id in done_ids:
                continue

            t0 = time.time()

            # Ground truth
            true_row    = labels_df[labels_df["spec"].astype(str) == spec_id]
            true_smiles = true_row["smiles"].iloc[0] if not true_row.empty else None
            true_key14  = canonical_inchikey14(true_smiles) if true_smiles else None
            true_fp     = morgan_fp(true_smiles) if true_smiles else None

            # Original predictions for this spectrum
            orig_row   = pred_df[pred_df["spec_id"].astype(str) == spec_id]
            if orig_row.empty:
                print(f"  [{batch_idx+1}/{n_test}] {spec_id}  NOT IN INPUT CSV, skipping")
                continue
            orig_row = orig_row.iloc[0]
            orig_smiles = [orig_row.get(c) for c in pred_cols]
            orig_smiles = [s if isinstance(s, str) else None for s in orig_smiles]

            # Move batch to device
            def to_dev(v):
                if hasattr(v, "to"):    return v.to(device)
                if isinstance(v, dict): return {k: to_dev(x) for k, x in v.items()}
                if isinstance(v, list): return [to_dev(x) for x in v]
                return v
            batch = to_dev(batch)

            # Encode spectrum
            try:
                _, y = encode_spectrum(model, batch, device)
            except Exception as e:
                print(f"  [{batch_idx+1}/{n_test}] {spec_id}  ENCODE ERROR: {e}")
                continue

            # Convert SMILES → graph tensors
            valid_graphs = []   # (idx_in_orig, X, E)
            for i, smi in enumerate(orig_smiles):
                X, E = smiles_to_dense(smi)
                if X is not None:
                    valid_graphs.append((i, X, E))

            # NLL scoring
            nll_scores = {i: float("nan") for i in range(n_samples)}
            if valid_graphs:
                try:
                    graphs_only = [(X, E) for _, X, E in valid_graphs]
                    X_batch, E_batch, node_mask = pad_graphs(graphs_only)
                    X_batch   = X_batch.to(device)
                    E_batch   = E_batch.to(device)
                    node_mask = node_mask.to(device)
                    y_rep     = y.expand(len(valid_graphs), -1).contiguous()

                    nlls = model.compute_mol_nll(
                        X_batch, E_batch, y_rep, node_mask, n_mc=args.nll_mc)

                    for rank, (orig_idx, _, _) in enumerate(valid_graphs):
                        nll_scores[orig_idx] = nlls[rank].item()
                except Exception as e:
                    print(f"  [{batch_idx+1}/{n_test}] {spec_id}  NLL ERROR (unranked): {e}")
                    if device.type == "cuda":
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass

            # Sort: valid molecules by NLL asc, invalid at end
            def sort_key(i):
                nll = nll_scores[i]
                return (math.isnan(nll), nll)

            order = sorted(range(n_samples), key=sort_key)
            reranked_smiles = [orig_smiles[i] for i in order]
            reranked_nlls   = [nll_scores[i]  for i in order]

            elapsed = time.time() - t0

            # Metrics on reranked list
            valid_smiles = [s for s in reranked_smiles if s]
            hits    = recall_at_k(reranked_smiles, true_key14) if true_key14 else {}
            max_tan = max_tanimoto(valid_smiles, true_fp)

            row = {
                "spec_id":      spec_id,
                "true_smiles":  true_smiles,
                "valid":        len(valid_smiles),
                "max_tanimoto": round(max_tan, 4),
                **{k: int(v) for k, v in hits.items()},
                "elapsed_s":    round(elapsed, 1),
            }
            for rank, (smi, nll_val) in enumerate(zip(reranked_smiles, reranked_nlls), 1):
                row[f"pred_{rank}"] = smi
                row[f"nll_{rank}"]  = round(nll_val, 4) if not math.isnan(nll_val) else None
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

            recall_str = "  ".join(f"{k}={'✓' if v else '✗'}" for k, v in hits.items())
            print(f"  [{batch_idx+1:>3}/{n_test}] {spec_id}  "
                  f"tan={max_tan:.3f}  {recall_str}  ({elapsed:.0f}s)")

    # ── Save & compare ────────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_csv, index=False)

    elapsed_total = time.time() - t_total
    W = 60

    # Reload original for comparison
    orig_df = pd.read_csv(in_csv)
    # Align on spec_id
    merged = results_df.merge(orig_df, on="spec_id", suffixes=("_new", "_old"))

    print(f"\n{'=' * W}")
    print(f"  Spectra reranked  : {len(results_df)}")
    print(f"  Total time        : {elapsed_total/60:.1f} min")
    print()
    print(f"  {'지표':<18} {'원본(무작위)':>14} {'NLL 재정렬':>12} {'변화':>10}")
    print(f"  {'-' * 56}")
    for m in ["recall@1","recall@5","recall@10","recall@20","max_tanimoto"]:
        if f"{m}_old" in merged.columns and f"{m}_new" in merged.columns:
            o = merged[f"{m}_old"].mean()
            n = merged[f"{m}_new"].mean()
            if "recall" in m:
                print(f"  {m:<18} {o*100:>13.1f}% {n*100:>11.1f}% {(n-o)*100:>+9.1f}pp")
            else:
                print(f"  {m:<18} {o:>14.4f} {n:>12.4f} {n-o:>+10.4f}")
    print(f"\n  Results saved: {out_csv}")
    print(f"{'=' * W}")


if __name__ == "__main__":
    main()
