# Pod bootstrap for the MiniCPM5 smoke (base-model bake-off gate 2). Pulls
# both Q4 GGUFs from HF, node for the sandbox, llama-cpp-python with CUDA:
# the prebuilt cu124 wheel on sm_86/sm_89 cards (3090/4090/A6000), a source
# build on sm_120 (5090; nvcc lives at /usr/local/cuda/bin, off PATH).
#   nohup bash pod_minicpm5_setup.sh > setup.log 2>&1 &
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
cd /workspace
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "gpu: $GPU"
python -c "import torch;x=torch.randn(512,512,device='cuda');print('cuda ok',(x@x).sum().item())"
pip install -q huggingface_hub
mkdir -p models
( hf download openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q4_K_M.gguf --local-dir /workspace/models > dl1.log 2>&1; echo "exit $?" >> dl1.log ) &
( hf download openbmb/MiniCPM5-2B-GGUF MiniCPM5-2B-Q4_K_M.gguf --local-dir /workspace/models > dl2.log 2>&1; echo "exit $?" >> dl2.log ) &
apt-get update -qq > /dev/null && apt-get install -y -qq nodejs > /dev/null && node --version
# Always a source build: the prebuilt cu124 wheel SIGILLs on some community
# hosts' CPUs (a 3090 box, 2026-09-08) and has no sm_120 kernels anyway.
export PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc
ARCH=86; echo "$GPU" | grep -q "4090" && ARCH=89; echo "$GPU" | grep -q "5090" && ARCH=120
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$ARCH" pip install --no-cache-dir llama-cpp-python==0.3.35 > build.log 2>&1
python -c "import llama_cpp; from llama_cpp import llama_cpp as C; print('llama_cpp', llama_cpp.__version__, 'gpu offload:', C.llama_supports_gpu_offload())"
wait
ls -l models/
echo "=== pod setup done ==="
