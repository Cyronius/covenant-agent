# Pod bootstrap for the Qwen3.8-27B runs (reactive-execution.md §7 step 1+).
# One-time per pod: verify CUDA, build llama-cpp-python against it, pull the
# GGUF straight from HF (12 GB; nothing that size leaves the dev box), node
# for the sandbox. Run detached: nohup bash pod_27b_setup.sh > setup.log 2>&1 &
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
cd /workspace
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch;x=torch.randn(512,512,device='cuda');print('cuda ok',(x@x).sum().item())"
nvcc --version | tail -1 || echo "no nvcc"
df -h /workspace | tail -1
pip install -q huggingface_hub pytest
mkdir -p models
( hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-IQ3_S.gguf --local-dir /workspace/models > dl.log 2>&1; echo "download exit $?" >> dl.log ) &
apt-get update -qq > /dev/null && apt-get install -y -qq nodejs > /dev/null && node --version
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" pip install --no-cache-dir llama-cpp-python==0.3.35 > build.log 2>&1 || {
  echo "source build failed, trying prebuilt cu124 wheel"; tail -20 build.log
  pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 > build2.log 2>&1
}
python -c "import llama_cpp; print('llama_cpp', llama_cpp.__version__)"
wait
ls -l models/
echo "=== pod setup done ==="
