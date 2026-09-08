#!/bin/bash
# Jean Zay: train one game per array task with the shipped recipe (config_default.json:
# per-frame bg->black recolor built from the training frames, 2x2 max-pool downscale,
# z_bg 5, pixel MSE, 100 epochs). The palette is built by the job from $SCRATCH/dataset/images/train
# (pixel statistics only, no labels) and stored as weights/<Game>/palette.json, so DLPInference
# applies the identical recolor and downscale. Runs take ~2h40 per game on an H100.
#
# Submit from a LOGIN shell (or rely on the WORK/SCRATCH fallbacks below):
#   /usr/bin/sbatch --array=0-$(($(wc -l < games.txt) - 1)) hpc_train.sbatch

#SBATCH --job-name=dlp_train
#SBATCH --constraint=h100
#SBATCH --account=ayj@h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out

module purge
export WORK="${WORK:-/lustre/fswork/projects/rech/ayj/ube83qw}"
export SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/ayj/ube83qw}"
export PATH="$WORK/envs/atari_det/bin:$PATH"
export MPLCONFIGDIR="$WORK/.cache/matplotlib"
python -c "import torch, sys; print('python', sys.executable, 'torch', torch.__version__)" || { echo "atari_det env not found"; exit 1; }

ROOT="$SCRATCH/dataset"

cd "$WORK/Atari-Object-Discovery-DLP"
GAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" games.txt)
echo "array task $SLURM_ARRAY_TASK_ID -> game $GAME on $(hostname)"

srun python -m dlp_inference.train --game "$GAME" --root "$ROOT" \
    --out "weights/$GAME"
