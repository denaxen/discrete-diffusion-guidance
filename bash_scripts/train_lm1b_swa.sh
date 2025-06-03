#!/bin/bash

# Setup environment
# cd ../ || exit  # Go to the root directory of the repo
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1

# Expecting:
#  - MODEL (ar, mdlm, udlm)
if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi
RUN_NAME="${MODEL} swa"

if [ "${MODEL}" = "ar" ]; then
  # AR
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="ar"
  T=0
  TIME_COND=False
  ZERO_RECON_LOSS=False
  sampling_use_cache=False
elif [ "${MODEL}" = "mdlm" ]; then
  # MDLM
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="subs"
  T=0
  TIME_COND=False
  ZERO_RECON_LOSS=False
  sampling_use_cache=True
elif [ "${MODEL}" = "udlm" ]; then
  # UDLM
  DIFFUSION="uniform"
  PARAMETERIZATION="d3pm"
  T=0
  TIME_COND=True
  ZERO_RECON_LOSS=True
  sampling_use_cache=False
else
  echo "MODEL must be one of ar, mdlm, udlm"
  exit 1
fi

python -u -m main \
  diffusion="${DIFFUSION}" \
  parameterization="${PARAMETERIZATION}" \
  T=${T} \
  time_conditioning=${TIME_COND} \
  zero_recon_loss=${ZERO_RECON_LOSS} \
  data="lm1b" \
  data.wrap=False \
  data.tokenizer_name_or_path=gpt2-large \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=1024 \
  loader.batch_size=64 \
  loader.eval_batch_size=128 \
  backbone="dit" \
  model=small \
  model.length=128 \
  optim.lr=3e-4 \
  training.guidance=null \
  training.compute_loss_on_pad_tokens=False \
  weights_averaging=swa \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
  trainer.log_every_n_steps=100 \
  trainer.max_steps=10_000 \
  trainer.precision=16-mixed \
  trainer.val_check_interval=10_000 \
  eval.generate_samples=True \
  sampling.num_sample_batches=1 \
  sampling.batch_size=2 \
  sampling.use_cache=${sampling_use_cache} \
  sampling.steps=128 \
  wandb.name="lm1b_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/lm1b/${RUN_NAME}"