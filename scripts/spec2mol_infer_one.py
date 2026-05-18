import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

# Must be set before torch imports — enables CPU fallback for MPS-unsupported ops (e.g. eigh)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from src.analysis.visualization import MolecularVisualization
from src.datasets import spec2mol_dataset
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete

PROTON_MASS = 1.007276
PPM_TOLERANCE = 5


def detect_device(preference: str) -> torch.device:
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    # auto: CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_parentmass(spec_file: Path) -> float:
    with spec_file.open() as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(">parentmass"):
                return float(line.split()[1])
    raise ValueError(f"No >parentmass entry found in {spec_file}")


def mol_to_smiles(mol) -> Optional[str]:
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def mass_match_fields(smiles, parentmass):
    empty = {"exact_mass": None, "mh_adduct_mass": None, "parentmass": parentmass,
             "ppm_error": None, "within_5ppm_mh": False}
    if not smiles:
        return empty
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return empty
    exact_mass = Descriptors.ExactMolWt(mol)
    mh_adduct_mass = exact_mass + PROTON_MASS
    ppm_error = abs(mh_adduct_mass - parentmass) / parentmass * 1e6
    return {
        "exact_mass": round(exact_mass, 4),
        "mh_adduct_mass": round(mh_adduct_mass, 4),
        "parentmass": parentmass,
        "ppm_error": round(ppm_error, 2),
        "within_5ppm_mh": ppm_error <= PPM_TOLERANCE,
    }


def move_to_device(value, device):
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def build_tiny_split(project_dir: Path, spec_id: str, work_dir: Path):
    labels_path = project_dir / "data" / "msg" / "labels.tsv"
    labels = pd.read_csv(labels_path, sep="\t").astype(str)

    if spec_id not in labels["spec"].values:
        raise ValueError(f"spec_id '{spec_id}' not found in {labels_path}")

    # Pick 2 helper entries (train/val) that are different from target
    helper_ids = [sid for sid in labels["spec"].tolist() if sid != spec_id][:2]
    if len(helper_ids) < 2:
        raise ValueError("labels.tsv has fewer than 3 entries — cannot build split.")

    selected_ids = helper_ids + [spec_id]
    subset = labels[labels["spec"].isin(selected_ids)].copy()

    work_dir.mkdir(parents=True, exist_ok=True)
    demo_labels = work_dir / "labels.tsv"
    demo_split = work_dir / "split.tsv"
    subset.to_csv(demo_labels, sep="\t", index=False)

    with demo_split.open("w", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["name", "split"])
        writer.writerow([helper_ids[0], "train"])
        writer.writerow([helper_ids[1], "val"])
        writer.writerow([spec_id, "test"])

    return demo_labels, demo_split


def load_model(cfg, checkpoint_path, dataset_infos, train_metrics, visualization_tools,
               extra_features, domain_features):
    model = Spec2MolDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        visualization_tools=visualization_tools,
        extra_features=extra_features,
        domain_features=domain_features,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys  : {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description="Single-spectrum inference with DiffMS (MSG dataset)")
    parser.add_argument("--spec-id", default="MassSpecGymID0046618",
                        help="Spectrum ID (must exist in data/msg/labels.tsv)")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of candidate molecules to generate")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoints/diffms_msg.ckpt",
                        help="Checkpoint path relative to project root")
    parser.add_argument("--out", default="outputs/spec2mol_one",
                        help="Output directory (relative to project root)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                        help="Device: auto selects CUDA > MPS > CPU")
    parser.add_argument("--max-attempts-per-sample", type=int, default=3,
                        help="Max sampling retries per candidate on error")
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    device = detect_device(args.device)
    print(f"Device : {device}")

    project_dir = PROJECT_DIR
    ckpt_path = (project_dir / args.checkpoint).resolve()
    out_dir = (project_dir / args.out).resolve()
    spec_file = project_dir / "data" / "msg" / "spec_files" / f"{args.spec_id}.ms"

    if not spec_file.exists():
        raise FileNotFoundError(f"Spectrum file not found: {spec_file}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    parentmass = parse_parentmass(spec_file)
    print(f"Spectrum   : {args.spec_id}")
    print(f"Parent mass: {parentmass}")

    print("Building dataset split...")
    demo_labels, demo_split = build_tiny_split(project_dir, args.spec_id, out_dir / "tmp")

    print("Loading config...")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(project_dir / "configs")):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=msg",
                "general.name=infer_one",
                "general.wandb=disabled",
                "general.gpus=0",
                "model.encoder_hidden_dim=512",
                "model.encoder_magma_modulo=2048",
                "train.eval_batch_size=1",
                "train.num_workers=0",
                f"dataset.datadir={project_dir / 'data' / 'msg'}",
                f"dataset.labels_file={demo_labels}",
                f"dataset.split_file={demo_split}",
                f"dataset.spec_folder={project_dir / 'data' / 'msg' / 'spec_files'}",
                f"dataset.subform_folder={project_dir / 'data' / 'msg' / 'subformulae' / 'default_subformulae'}",
            ],
        )

    print("Initializing dataset...")
    datamodule = spec2mol_dataset.Spec2MolDataModule(cfg)
    dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    extra_features = (
        ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        if cfg.model.extra_features
        else DummyExtraFeatures()
    )
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, extra_features=extra_features, domain_features=domain_features
    )
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    print(f"Loading model from {ckpt_path.name}...")
    model = load_model(cfg, ckpt_path, dataset_infos, train_metrics, visualization_tools,
                       extra_features, domain_features)
    model = model.to(device)

    print(f"Encoding spectrum and sampling {args.num_samples} candidates...")
    batch = move_to_device(next(iter(datamodule.test_dataloader())), device)

    with torch.no_grad():
        output, aux = model.encoder(batch)
        data = batch["graph"]
        if model.merge == "mist_fp":
            data.y = aux["int_preds"][-1]
        elif model.merge in {"merge-encoder_output-linear", "merge-encoder_output-mlp"}:
            data.y = model.merge_function(aux["h0"])
        elif model.merge == "downproject_4096":
            data.y = model.merge_function(output)

        rows = []
        for i in range(args.num_samples):
            print(f"  [{i + 1:>{len(str(args.num_samples))}}/{args.num_samples}] sampling...", end="\r")
            mol, error = None, None
            for _ in range(args.max_attempts_per_sample):
                try:
                    mol = model.sample_batch(data)[0]
                    break
                except Exception as e:
                    error = str(e)
            smiles = mol_to_smiles(mol)
            row = {"rank": i + 1, "smiles": smiles}
            row.update(mass_match_fields(smiles, parentmass))
            row["sampling_error"] = error
            rows.append(row)
        print()

    # True SMILES from labels
    labels_df = pd.read_csv(demo_labels, sep="\t")
    true_smiles_series = labels_df.loc[labels_df["spec"] == args.spec_id, "smiles"]
    true_smiles = true_smiles_series.iloc[0] if not true_smiles_series.empty else "N/A"

    # Save CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.spec_id}_predictions.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Pretty-print results
    W = 72
    print(f"\n{'=' * W}")
    print(f"  Spectrum  : {args.spec_id}")
    print(f"  Parent m/z: {parentmass}  ([M+H]+ tolerance: {PPM_TOLERANCE} ppm)")
    print(f"  True SMILES: {true_smiles}")
    print(f"{'=' * W}")
    print(f"{'Rank':<5} {'OK':>3} {'≤5ppm':>6} {'PPM err':>8}  SMILES")
    print(f"{'-' * W}")
    for row in rows:
        valid_mark = "✓" if row["smiles"] else "✗"
        ppm_mark   = "✓" if row["within_5ppm_mh"] else ("✗" if row["smiles"] else "-")
        ppm_str    = f"{row['ppm_error']:.1f}" if row["ppm_error"] is not None else "-"
        smi_str    = (row["smiles"] or "(invalid)")[:55]
        print(f"{row['rank']:<5} {valid_mark:>3} {ppm_mark:>6} {ppm_str:>8}  {smi_str}")
    print(f"{'-' * W}")
    valid_n  = sum(1 for r in rows if r["smiles"])
    ppm_n    = sum(1 for r in rows if r["within_5ppm_mh"])
    print(f"  Valid: {valid_n}/{args.num_samples}   Within 5 ppm: {ppm_n}/{args.num_samples}")
    print(f"  Saved: {out_csv}")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()
