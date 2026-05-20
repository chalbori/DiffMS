"""
Batch inference over data/sop test spectra.

For each test spectrum:
  - Encode spectrum once with the MIST encoder
  - Replicate the graph n_samples times → one sample_batch() call (batched denoising)
  - Collect top-k SMILES, compute Tanimoto similarity vs. ground truth
  - Report top-1/5/10/20 hit rate + mean max-Tanimoto

Usage:
    python scripts/infer_sop.py [--num-samples 20] [--device auto] [--out outputs/sop_results]

Runtime estimate (MPS, n_samples=20):
    ~296 denoising calls × ~30 s ≈ ~2.5 h
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
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

RDLogger.DisableLog("rdApp.*")


# ── Device ───────────────────────────────────────────────────────────────────

def detect_device(preference: str) -> torch.device:
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Molecule utilities ───────────────────────────────────────────────────────

def mol_to_smiles(mol) -> Optional[str]:
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def canonical_inchikey14(smiles: str) -> Optional[str]:
    """First 14 chars of InChIKey = connectivity layer (ignores stereochemistry)."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    from rdkit.Chem.inchi import MolToInchiKey
    key = MolToInchiKey(mol)
    return key[:14] if key else None


def morgan_fp(smiles: str, radius: int = 2, nbits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ── Batched sampling ─────────────────────────────────────────────────────────

def encode_spectrum(model, batch, device):
    """Run encoder and return (data_graph, y_vec)."""
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
    return data, y   # data: Batch(bs=1),  y: (1, dim)


def sample_batch_repeated(model, data, y, n_samples: int, device):
    """
    Generate n_samples molecules from a single spectrum in one batched call.

    Replicates the graph n_samples times so the 500-step denoising loop
    processes all samples in parallel instead of sequentially.

    Returns:
        mols: list of n_samples RDKit molecules
        X_onehot: (n_samples, n, dx) one-hot atom types
        E_onehot: (n_samples, n, n, de) one-hot edge types
        node_mask: (n_samples, n) bool
        y_rep: (n_samples, dy) spectrum conditioning vector
    """
    single = data.get_example(0)
    data_list = [single.clone() for _ in range(n_samples)]
    batch_rep = Batch.from_data_list(data_list).to(device)
    y_rep = y.expand(n_samples, -1).contiguous()
    batch_rep.y = y_rep
    mols, X_onehot, E_onehot, node_mask = model.sample_batch(batch_rep, return_graphs=True)
    return mols, X_onehot, E_onehot, node_mask, y_rep


# ── Model loading ─────────────────────────────────────────────────────────────

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


# ── Metrics ──────────────────────────────────────────────────────────────────

def recall_at_k(pred_smiles_list, true_key14: str, ks=(1, 5, 10, 20)) -> dict:
    result = {}
    for k in ks:
        result[f"recall@{k}"] = False
        for smi in pred_smiles_list[:k]:
            if canonical_inchikey14(smi) == true_key14:
                result[f"recall@{k}"] = True
                break
    return result


def max_tanimoto(pred_smiles_list, true_fp) -> float:
    return max((tanimoto(morgan_fp(s), true_fp) for s in pred_smiles_list if s), default=0.0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch SOP inference with DiffMS")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Candidate molecules per spectrum")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoints/diffms_msg.ckpt")
    parser.add_argument("--out", default="outputs/sop_results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Retries on sampling error (per spectrum)")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Limit number of test spectra (for debugging)")
    parser.add_argument("--nll-mc", type=int, default=5,
                        help="Monte Carlo samples for NLL scoring (higher = more accurate, slower)")
    args = parser.parse_args()

    device   = detect_device(args.device)
    ckpt_path = (PROJECT_DIR / args.checkpoint).resolve()
    out_dir   = (PROJECT_DIR / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Device     : {device}")
    print(f"Checkpoint : {ckpt_path.name}")
    print(f"Samples/spec: {args.num_samples}")

    # ── Config ────────────────────────────────────────────────────────────────
    print("\nLoading config & dataset...")
    sop_dir = PROJECT_DIR / "data" / "sop"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_DIR / "configs")):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=sop",
                "general.name=infer_sop",
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

    # ── Dataset ───────────────────────────────────────────────────────────────
    datamodule   = spec2mol_dataset.Spec2MolDataModule(cfg)
    dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    extra_features = (
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
    print(f"Loading model...")
    model = load_model(cfg, ckpt_path, dataset_infos, train_metrics,
                       visualization_tools, extra_features, domain_features)
    model = model.to(device)

    # ── Ground-truth labels ───────────────────────────────────────────────────
    labels_df = pd.read_csv(PROJECT_DIR / "data" / "sop" / "labels.tsv", sep="\t")
    test_loader = datamodule.test_dataloader()
    n_test = len(test_loader)
    if args.max_test:
        n_test = min(n_test, args.max_test)
    print(f"\nTest spectra: {n_test}")

    # ── Resume support ────────────────────────────────────────────────────────
    out_csv = out_dir / "predictions.csv"
    progress_file = out_dir / "progress.txt"
    rows = []
    done_ids: set = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done_ids = set(prev["spec_id"].astype(str))
        print(f"Resuming: {len(done_ids)} spectra already done, skipping them.\n")
    else:
        print(f"Starting inference...\n")

    spec_idx = 0
    run_start = time.time()


    def write_progress(done, total, rows, elapsed_last):
        elapsed_total = time.time() - run_start
        avg_s = elapsed_total / done if done else 0
        eta_s = avg_s * (total - done)
        eta_h, eta_m = divmod(int(eta_s), 3600)
        eta_m //= 60

        df = pd.DataFrame(rows)
        recall_cols = [c for c in df.columns if c.startswith("recall@")]
        recall_lines = "".join(
            f"  {col:<14}: {df[col].mean() * 100:.1f}%\n" for col in recall_cols
        ) if recall_cols else "  (no ground truth yet)\n"

        with open(progress_file, "w") as f:
            f.write(f"Progress  : {done}/{total}  ({done/total*100:.1f}%)\n")
            f.write(f"Elapsed   : {elapsed_total/3600:.1f}h  ({avg_s:.0f}s/spec)\n")
            f.write(f"ETA       : {eta_h}h {eta_m:02d}m\n")
            f.write(f"Last spec : {elapsed_last:.0f}s\n")
            f.write(f"---\n")
            if len(df):
                f.write(f"SMILES validity (mean per spectrum): {df['valid'].mean() / args.num_samples * 100:.1f}%\n")
                f.write(f"Max Tan   : {df['max_tanimoto'].mean():.4f}\n")
            f.write(recall_lines)

    with torch.no_grad():
        for batch in (x for i, x in enumerate(test_loader) if i < n_test):
            spec_idx += 1
            # batch["names"] is populated by the PeakFormula collate fn
            spec_id = str(batch["names"][0])

            if spec_id in done_ids:
                continue

            true_row = labels_df[labels_df["spec"].astype(str) == spec_id]
            true_smiles = true_row["smiles"].iloc[0] if not true_row.empty else None
            true_key14  = canonical_inchikey14(true_smiles) if true_smiles else None
            true_fp     = morgan_fp(true_smiles) if true_smiles else None

            t0 = time.time()

            # Move batch to device
            def to_dev(v):
                if hasattr(v, "to"):    return v.to(device)
                if isinstance(v, dict): return {k: to_dev(x) for k, x in v.items()}
                if isinstance(v, list): return [to_dev(x) for x in v]
                return v
            batch = to_dev(batch)

            # Encode
            try:
                data, y = encode_spectrum(model, batch, device)
            except Exception as e:
                print(f"  [{spec_idx}/{n_test}] {spec_id}  ENCODE ERROR: {e}")
                continue

            # Sample (with retry on error)
            mols, X_onehot, E_onehot, node_mask_rep, y_rep = [], None, None, None, None
            for attempt in range(args.max_attempts):
                try:
                    mols, X_onehot, E_onehot, node_mask_rep, y_rep = \
                        sample_batch_repeated(model, data, y, args.num_samples, device)
                    break
                except Exception as e:
                    if attempt == args.max_attempts - 1:
                        print(f"  [{spec_idx}/{n_test}] {spec_id}  SAMPLE ERROR: {e}")

            # NLL scoring: rank candidates by model likelihood (lower NLL = more likely)
            nll_values = [float("nan")] * len(mols)
            order = list(range(len(mols)))
            if mols and X_onehot is not None:
                try:
                    nlls = model.compute_mol_nll(
                        X_onehot, E_onehot, y_rep, node_mask_rep, n_mc=args.nll_mc)
                    order = nlls.argsort().tolist()
                    nll_values = nlls.tolist()
                except Exception as e:
                    print(f"  [{spec_idx}/{n_test}] {spec_id}  NLL ERROR (unranked): {e}")
                    # Sync CUDA to surface any device-side errors and clear them
                    if device.type == "cuda":
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass

            elapsed = time.time() - t0

            # Reorder molecules by NLL (best first)
            mols_ranked = [mols[i] for i in order]
            nll_ranked  = [nll_values[i] for i in order]

            # Convert to SMILES
            pred_smiles = [mol_to_smiles(m) for m in mols_ranked]
            valid_smiles = [s for s in pred_smiles if s]

            # Metrics
            hits      = recall_at_k(valid_smiles, true_key14) if true_key14 else {}
            max_tan   = max_tanimoto(valid_smiles, true_fp)
            valid_n   = len(valid_smiles)

            row = {
                "spec_id":      spec_id,
                "true_smiles":  true_smiles,
                "valid":        valid_n,
                "max_tanimoto": round(max_tan, 4),
                **{k: int(v) for k, v in hits.items()},
                "elapsed_s":    round(elapsed, 1),
            }
            # Store predictions in NLL-ranked order with their scores
            for rank, (smi, nll_val) in enumerate(zip(pred_smiles, nll_ranked), 1):
                row[f"pred_{rank}"] = smi
                row[f"nll_{rank}"]  = round(nll_val, 4) if not math.isnan(nll_val) else None
            rows.append(row)

            recall_str = "  ".join(f"{k}={'✓' if v else '✗'}" for k, v in hits.items())
            print(f"  [{spec_idx:>3}/{n_test}] {spec_id}  "
                  f"valid={valid_n}/{args.num_samples}  "
                  f"tan={max_tan:.3f}  {recall_str}  ({elapsed:.0f}s)")

            # Update progress file and partial CSV after every spectrum
            write_progress(spec_idx, n_test, rows, elapsed)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results_df)
    recall_cols = [c for c in results_df.columns if c.startswith("recall@")]
    W = 56
    print(f"\n{'=' * W}")
    print(f"  Spectra evaluated : {n}")
    print(f"  Validity (mean)   : {results_df['valid'].mean() / args.num_samples * 100:.1f}%")
    print(f"  Max Tanimoto (mean): {results_df['max_tanimoto'].mean():.4f}")
    if recall_cols:
        for col in recall_cols:
            rate = results_df[col].mean() * 100
            print(f"  {col:<18}: {rate:.1f}%")
    print(f"  Results saved     : {out_csv}")
    print(f"{'=' * W}")


if __name__ == "__main__":
    main()
