# Stage 2: the runpod torch image ships no nvcc, and the prebuilt cu124
# wheels have no sm_120 (RTX 5090) kernels, so install the toolkit and build
# llama-cpp-python against it. Runs after pod_27b_setup.sh's fallback.
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
cd /workspace
while ! grep -q "llama_cpp\|pod setup done\|Error\|error" setup.log 2>/dev/null; do sleep 15; done
apt-get install -y -qq cuda-toolkit-12-8 > apt_cuda.log 2>&1 || apt-get install -y -qq cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 >> apt_cuda.log 2>&1
export PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH
export CUDACXX=$(command -v nvcc)
nvcc --version | tail -1
pip uninstall -y -q llama-cpp-python || true
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120" pip install --no-cache-dir llama-cpp-python==0.3.35 > build_cuda.log 2>&1
python -c "import llama_cpp; print('llama_cpp', llama_cpp.__version__); from llama_cpp import llama_cpp as C; print('gpu offload supported:', C.llama_supports_gpu_offload())"
echo "=== stage 2 done ==="
