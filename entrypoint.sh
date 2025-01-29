#!/bin/bash

# Download models if not already present
MODEL_1_5B="${MODEL_DIR}/deepseek-r1-distill-qwen-1.5b"
MODEL_7B="${MODEL_DIR}/deepseek-r1-distill-qwen-7b"

# Download 1.5B model files
if [ ! -d "${MODEL_1_5B}" ]; then
    echo "Downloading 1.5B model..."
    mkdir -p "${MODEL_1_5B}"
    wget --header="Authorization: Bearer YOUR_HUGGINGFACE_TOKEN" -P "${MODEL_1_5B}" \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/main/model.safetensors \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/main/config.json \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/main/tokenizer.json \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/main/tokenizer_config.json \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/main/generation_config.json \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/main/model.safetensors.index.json
fi

# Download 7B model files
if [ ! -d "${MODEL_7B}" ]; then
    echo "Downloading 7B model..."
    mkdir -p "${MODEL_7B}"
    wget --header="Authorization: Bearer YOUR_HUGGINGFACE_TOKEN" -P "${MODEL_7B}" \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/main/model-00001-of-000002.safetensors \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/main/model-00002-of-000002.safetensors \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/main/config.json \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/main/tokenizer.json \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/main/tokenizer_config.json \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/main/generation_config.json \
        https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/main/model.safetensors.index.json
fi

# Start the server
python3 safetensors_endpoint.py 