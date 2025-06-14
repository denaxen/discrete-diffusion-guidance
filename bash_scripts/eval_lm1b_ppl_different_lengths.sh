export HYDRA_FULL_ERROR=1
export NCCL_P2P_LEVEL=NVL

# Expecting:
#  - MODEL (choices: ar, mdlm, udlm)
#  - SEED (optional: default = 1)

if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi
if [ -z "${SEED}" ]; then
  SEED=1
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

# shellcheck disable=SC2086
python -u -m main \
    hydra.output_subdir=null \
    hydra.run.dir="${PWD}" \
    seed=${SEED} \
    mode="lengths_eval" \
    eval.checkpoint_path="${CKPT}/checkpoints/last.ckpt" \
    eval.generate_samples=False \
    +eval.lengths="[8, 16, 32, 64, 128]" \
    loader.eval_batch_size=${BATCH_SIZE} \
    data=lm1b \
    data.tokenizer_name_or_path=gpt2-large \
    data.wrap=False \
    backbone=dit \
    model=small \
    model.length=128 \
    training.guidance=null \
    parameterization=${PARAMETERIZATION} \
    diffusion=${DIFFUSION} \
    time_conditioning=${TIME_COND} \
    zero_recon_loss=${ZERO_RECON_LOSS} \
    T=${TRAIN_T}
