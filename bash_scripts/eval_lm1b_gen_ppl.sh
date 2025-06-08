# Setup environment
# cd ../ || exit  # Go to the root directory of the repo

export HYDRA_FULL_ERROR=1
export NCCL_P2P_LEVEL=NVL

# Expecting:
#  - MODEL (choices: ar, mdlm, udlm)
#  - SAMPLING_STEPS (optional: default = 128)
#  - SEED (optional: default = 1)
#  - USE_FLOAT64 (optional: default = False)
#  - CHECKPOINT (optional: default = last.ckpt)

# Get checkpoint name from first argument, default to last.ckpt
CHECKPOINT="${1:-last.ckpt}"

if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi
if [ -z "${SAMPLING_STEPS}" ]; then
  SAMPLING_STEPS=128
fi
if [ -z "${SEED}" ]; then
  SEED=1
fi
if [ -z "${USE_FLOAT64}" ]; then
  USE_FLOAT64=False
fi

if [[ "${MODEL}" == *"ar"* ]]; then
  parameterization="ar"
  diffusion="absorbing_state"
  TRAIN_T=0
  time_conditioning=False
  sampling_use_cache=False
  CKPT="${PWD}/outputs/lm1b/${MODEL}"
elif [[ "${MODEL}" == *"mdlm"* ]]; then
  parameterization="subs"
  diffusion="absorbing_state"
  TRAIN_T=0
  time_conditioning=False
  sampling_use_cache=True
  CKPT="${PWD}/outputs/lm1b/${MODEL}"
elif [[ "${MODEL}" == *"udlm"* ]]; then
  parameterization="d3pm"
  diffusion="uniform"
  TRAIN_T=0
  time_conditioning=True
  sampling_use_cache=False
  CKPT="${PWD}/outputs/lm1b/${MODEL}"
else
  echo "Invalid MODEL: ${MODEL}"
  exit 1
fi
generated_seqs_path="${CKPT}/samples-lm1b-gen-ppl-eval-float64-${USE_FLOAT64}_add-CLS_T-${SAMPLING_STEPS}_seed-${SEED}.json"

# shellcheck disable=SC2086
python -u -m main \
    hydra.output_subdir=null \
    "hydra.run.dir=${CKPT}" \
    seed=${SEED} \
    mode="gen_ppl_eval" \
    "eval.checkpoint_path=${CKPT}/checkpoints/${CHECKPOINT}" \
    data=lm1b \
    data.tokenizer_name_or_path=gpt2-large \
    backbone=dit \
    model=small \
    model.length=128 \
    training.guidance=null \
    parameterization=${parameterization} \
    diffusion=${diffusion} \
    time_conditioning=${time_conditioning} \
    T=${TRAIN_T} \
    sampling.num_sample_batches=32 \
    sampling.batch_size=32 \
    sampling.steps=${SAMPLING_STEPS} \
    sampling.use_cache=${sampling_use_cache} \
    sampling.use_float64=${USE_FLOAT64} \
    "eval.generated_samples_path=${generated_seqs_path}" \
    +eval.generative_ppl_model_name_or_path="gpt2-large" \
    wandb.job_type="get_ppl" \
    "wandb.name=lm1b_${MODEL}_gen_ppl"
