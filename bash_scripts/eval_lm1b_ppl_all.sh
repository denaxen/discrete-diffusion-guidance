# Setup environment
# cd ../ || exit  # Go to the root directory of the repo

export HYDRA_FULL_ERROR=1
export NCCL_P2P_LEVEL=NVL
TRAIN_T=0
CKPT="${PWD}/outputs/lm1b/ppl_all"
SEED=1

# shellcheck disable=SC2086
python -u -m main \
    hydra.output_subdir=null \
    "hydra.run.dir=${CKPT}" \
    seed=${SEED} \
    mode="ppl_eval_all" \
    "eval.checkpoint_path=${CKPT}" \
    data=lm1b \
    data.tokenizer_name_or_path=gpt2-large \
    data.wrap=False \
    backbone=dit \
    model=small \
    model.length=128 \
    training.guidance=null \
    T=${TRAIN_T} \
    eval.generate_samples=False \
    wandb.job_type="get_ppl" \
    "wandb.name=lm1b_ppl_all"
