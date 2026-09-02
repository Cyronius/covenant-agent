from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3.5-0.8B", allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"])
print("downloaded to", p)
