bash setup_all.sh
# locate libcuda.so.1
# sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
# sudo ldconfig
export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu/libcuda.so.1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export MODEL=udlm
bash bash_scripts/train_lm1b_swa.sh
