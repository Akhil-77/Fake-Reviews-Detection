# Sol HPC Scripts

How to train on ASU Sol once you have a session.

## One-time setup (do this once after cloning)

```bash
cd /scratch/aabhilas/Fake-Reviews-Detection
bash scripts/setup_env.sh
```

This creates a conda env `fakereview` with PyTorch (CUDA 12.1), TensorFlow, transformers, etc. ~5 min.

## Get the dataset

Place `yelpzip.csv` in the project root. Expected columns: `text`, `label` (`fake`/`real` or `-1`/`1`).

If you only have raw YelpZIP files (`metadata.txt` + `reviewContent.txt`), you'll need a converter — ping Claude to write one.

## Run training

**Option A — fire-and-forget (recommended)**: one job runs the whole pipeline
```bash
sbatch scripts/run_all.sbatch
```
Wall time: ~3-4 hours on A100. Logs in `logs/all_<jobid>.out`.

**Option B — staged** (more control, easier to retry one step):
```bash
sbatch scripts/run_data_prep.sbatch        # ~10 min, CPU only
sbatch scripts/run_classical.sbatch        # ~30 min, CPU only
sbatch scripts/run_deep.sbatch             # ~30 min, A100
sbatch scripts/run_transformers.sbatch     # ~90 min, A100
```

Each later step depends on `data_prep` artifacts, so submit `run_data_prep` first and wait for it before submitting the rest. Or use SLURM dependencies:
```bash
JOB1=$(sbatch --parsable scripts/run_data_prep.sbatch)
sbatch --dependency=afterok:$JOB1 scripts/run_classical.sbatch
sbatch --dependency=afterok:$JOB1 scripts/run_deep.sbatch
sbatch --dependency=afterok:$JOB1 scripts/run_transformers.sbatch
```

## Monitoring

```bash
squeue -u $USER                       # see your queued/running jobs
tail -f logs/all_<jobid>.out          # follow log in real time
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS  # post-mortem
```

## If `--qos=class` doesn't work

The QOS depends on what your account is allowed. If submission fails with "Invalid qos", edit each `.sbatch` file and change the `--qos` line to whatever your OnDemand form's QOS dropdown showed (e.g. `--qos=public`, `--qos=grp_xyz`).
