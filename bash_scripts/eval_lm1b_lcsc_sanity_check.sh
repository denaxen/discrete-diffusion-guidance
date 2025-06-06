#!/bin/bash

# TODO: increase checkpoint interval to 10k in large scale experiments

# Setup environment
# cd ../ || exit  # Go to the root directory of the repo
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1
export SEED=42

# Expecting:
#  - MODEL (ar, mdlm, udlm)
if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi

if [ "${MODEL}"  = "ar" ]; then
  PARAMETERIZATION="ar"
  DIFFUSION="absorbing_state"
  TRAIN_T=0
  ZERO_RECON_LOSS=False
  TIME_COND=False
  BATCH_SIZE=128
  CKPT="${PWD}/outputs/lm1b/ar"
elif [ "${MODEL}" = "mdlm" ]; then
  PARAMETERIZATION="subs"
  DIFFUSION="absorbing_state"
  TRAIN_T=0
  ZERO_RECON_LOSS=False
  TIME_COND=False
  BATCH_SIZE=128
  CKPT="${PWD}/outputs/lm1b/mdlm"
elif [ "${MODEL}" = "udlm" ]; then
  PARAMETERIZATION="d3pm"
  DIFFUSION="uniform"
  TRAIN_T=0
  ZERO_RECON_LOSS=True
  TIME_COND=True
  BATCH_SIZE=64
  CKPT="${PWD}/outputs/lm1b/udlm"
else
  echo "Invalid MODEL: ${MODEL}"
  exit 1
fi

python -u -m main \
    hydra.output_subdir=null \
    hydra.run.dir="${CKPT}" \
    seed=${SEED} \
    mode="lcsc" \
    eval.checkpoint_path="${CKPT}/checkpoints/last.ckpt" \
    eval.generate_samples=False \
    loader.eval_batch_size=${BATCH_SIZE} \
    data=lm1b \
    data.wrap=False \
    data.tokenizer_name_or_path=gpt2-large \
    backbone=dit \
    model=small \
    model.length=128 \
    training.guidance=null \
    parameterization=${PARAMETERIZATION} \
    diffusion=${DIFFUSION} \
    time_conditioning=${TIME_COND} \
    zero_recon_loss=${ZERO_RECON_LOSS} \
    T=${TRAIN_T} \
    checkpointing.save_dir="${CKPT}" \
    +lcsc.metric="ppl" \
    +lcsc.output_ckpt="${CKPT}/merged_lcsc.ckpt" \
    +lcsc.population_size=1 \
    +lcsc.top_k=1 \
    +lcsc.iterations=1 \
    +lcsc.mutation_sigma=0.0 \
    +lcsc.offspring_per_iter=1 \
    +lcsc.num_sample_batches=1 \
    +eval.generative_ppl_model_name_or_path="gpt2-large" \
    wandb.job_type="lcsc" \
    wandb.name="lm1b_lcsc_${MODEL}"