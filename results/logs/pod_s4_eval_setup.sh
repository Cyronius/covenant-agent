# Pod bootstrap for the S4 typed-surface re-score (R6 §0). node for the
# sandbox, llama-cpp-python built from git main with CUDA: the 0.3.35 PyPI
# release does not know the qwen35 arch (R6 §4), and the prebuilt wheels
# have no sm_120 kernels for the 5090.
#   nohup bash pod_s4_eval_setup.sh > setup.log 2>&1 < /dev/null &
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
cd /workspace
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader); echo "gpu: $GPU"
python -c "import torch;x=torch.randn(512,512,device='cuda');print('cuda ok',(x@x).sum().item())"
( apt-get update -qq > apt.log 2>&1 && apt-get install -y -qq nodejs >> apt.log 2>&1 && node --version ) &
export PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc
nvcc --version | tail -1
ARCH=86; echo "$GPU" | grep -q "4090" && ARCH=89; echo "$GPU" | grep -q "5090" && ARCH=120
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$ARCH" pip install --no-cache-dir \
  "git+https://github.com/abetlen/llama-cpp-python.git@main" > build.log 2>&1
python -c "import llama_cpp; from llama_cpp import llama_cpp as C; print('llama_cpp', llama_cpp.__version__, 'gpu offload:', C.llama_supports_gpu_offload())"
wait
node --version
echo "=== pod setup done ==="
