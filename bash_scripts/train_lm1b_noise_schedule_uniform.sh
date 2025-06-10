#!/bin/bash

# TODO: increase checkpoint interval to 10k in large scale experiments

# Setup environment
# cd ../ || exit  # Go to the root directory of the repo
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1
NOISE_SCHEDULE_FRACTION="${1:-0.1}"
MAX_STEPS=10000


# Expecting:
#  - MODEL (ar, mdlm, udlm)
if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi
RUN_NAME="${MODEL} nsu ${NOISE_SCHEDULE_FRACTION}"

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
  training.noise_schedule_warmup=True \
  training.noise_schedule_warmup_fraction=${NOISE_SCHEDULE_FRACTION} \
  training.noise_schedule_uniform=True \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1_000 \
  trainer.log_every_n_steps=100 \
  trainer.max_steps=${MAX_STEPS} \
  trainer.precision=16-mixed \
  trainer.val_check_interval=10_000 \
  eval.generate_samples=True \
  sampling.num_sample_batches=1 \
  sampling.batch_size=2 \
  sampling.use_cache=${sampling_use_cache} \
  sampling.steps=128 \
  wandb.name="lm1b_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/lm1b/${RUN_NAME}"