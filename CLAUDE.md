# DiffMS — Project Guide for Claude Code

## What this project does
DiffMS is a diffusion model that generates molecular graphs (SMILES) conditioned on MS/MS spectra.
Architecture: MIST spectrum encoder → downproject_4096 merge → DiGress-style graph diffusion decoder (500 steps).

## Key files
- `src/spec2mol_main.py` — main entry point (training + test via PyTorch Lightning)
- `src/diffusion_model_spec2mol.py` — model definition; `sample_batch(data)` generates molecules
- `scripts/infer_sop.py` — **batch inference over custom SOP dataset** (main script to run)
- `scripts/prepare_sop_input.py` — converts `info_spectrum_sop.csv` → `data/sop/` pipeline files
- `scripts/spec2mol_infer_one.py` — single-spectrum inference (MSG dataset)

## Pretrained checkpoint
- Location: `checkpoints/checkpoints/diffms_msg.ckpt` (raw state dict, not full PL checkpoint)
- MSG large model: `encoder_hidden_dim=512`, `encoder_magma_modulo=2048`
- Download from Zenodo if missing: `https://zenodo.org/records/15122968`

## Datasets
| Dataset | Config | Data dir |
|---------|--------|----------|
| MSG (MassSpecGym) | `configs/dataset/msg.yaml` | `data/msg/` |
| MSG mini (3 spectra test) | `configs/dataset/msg_mini.yaml` | `data/msg/` |
| SOP (custom, 298 spectra) | `configs/dataset/sop.yaml` | `data/sop/` |

## SOP dataset pipeline
The SOP dataset comes from `info_spectrum_sop.csv` (proprietary, not in repo).

To regenerate `data/sop/` from the CSV:
```bash
PYTHONPATH=src python scripts/prepare_sop_input.py
```

Selection logic: per (c_id, prec_type) in {[M+H]+, [M+Na]+}, keep qualified=1 or highest TIC.
Result: 298 spectra from 196 unique compounds.

`data/sop/` is committed so regeneration is not required for inference.

## Running inference on SOP dataset

```bash
# Full run: 296 test spectra × 20 samples
PYTORCH_ENABLE_MPS_FALLBACK=1 WANDB_MODE=disabled PYTHONPATH=src \
python -u scripts/infer_sop.py \
  --num-samples 20 \
  --out outputs/sop_results \
  --device auto \
  > /tmp/infer_sop.log 2>&1 &

# Monitor progress
tail -f /tmp/infer_sop.log
```

On CUDA server, `--device auto` picks CUDA automatically. No `PYTORCH_ENABLE_MPS_FALLBACK` needed.

Expected runtime:
- MPS (Apple Silicon): ~43s/spectrum → ~3.5h for 296 spectra
- CUDA (A100): estimated ~5–10s/spectrum → ~0.5–1h

### Quick test (3 spectra only)
```bash
PYTHONPATH=src python -u scripts/infer_sop.py --num-samples 2 --max-test 3 --out outputs/sop_test
```

## Output format
`outputs/sop_results/predictions.csv` columns:
- `spec_id`, `true_smiles`, `valid` (count), `max_tanimoto`
- `hit@1`, `hit@5`, `hit@10`, `hit@20` (InChIKey-14 match)
- `pred_1` … `pred_20` (candidate SMILES)

## Mac MPS notes (not needed on CUDA server)
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before running (for `torch.linalg.eigh`)
- `gpus: 0` in config — MPS is auto-detected separately from CUDA
- `distributions.py` casts prob tensor to float32 to avoid MPS float64 error

## Environment setup
```bash
conda create -y -c conda-forge -n diffms rdkit=2024.09.4 python=3.9
conda activate diffms
pip install torch --index-url https://download.pytorch.org/whl/cu118  # adjust for CUDA version
pip install -e .
```

## Common issues
| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'models'` | Run with `PYTHONPATH=src` |
| `Cannot convert MPS Tensor to float64` | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Raw state dict checkpoint | Handled automatically — `_is_pl_checkpoint()` in `spec2mol_main.py` |
| WandB API key error | Set `WANDB_MODE=disabled` |
