# LeonardoGPU

Local GPU inference server for DeepSeek and other LLMs.

## Models Supported
- DeepSeek-R1-Distill-Qwen-7B (SafeTensors)
- DeepSeek-R1-Distill-Qwen-14B (GGUF)

## Endpoints
- /infer (SafeTensors) - Port 8000
- /infer (GGUF) - Port 8001

## Requirements
- CUDA-capable GPU
- Python 3.8+
- PyTorch with CUDA support
