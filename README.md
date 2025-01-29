# LeonardoGPU: High-Performance Local LLM Inference Server By Adrian Vargas

## Project Overview
LeonardoGPU is a high-performance inference server designed to run Large Language Models (LLMs) locally on GPU hardware. The project implements two distinct endpoints for different model formats, optimizing for both speed and memory efficiency.

## Technical Architecture

### 1. SafeTensors Endpoint (Port 8000)
- Implements HuggingFace Transformers for native PyTorch model loading
- Utilizes CUDA acceleration with FP16 precision
- Features optimized generation parameters:
  - Temperature control (0.3) for deterministic outputs
  - Top-p (0.7) and Top-k (50) sampling for focused responses
  - Repetition penalty (1.2) to prevent loops
  - N-gram repetition prevention
  - Length optimization with early stopping

### 2. GGUF Endpoint (Port 8001)
- Leverages llama.cpp for quantized model inference
- Implements GPU acceleration with configurable layer offloading
- Optimized batch processing for prompt evaluation
- Memory-efficient with 4-bit quantization support

## Performance Optimizations
- Model preloading at server startup
- Warm-up inference to prime GPU memory
- Automatic device mapping for optimal GPU utilization
- Low CPU memory usage configurations
- Response post-processing for coherent output

## Models Supported
1. DeepSeek-R1-Distill-Qwen-7B (SafeTensors)
   - 7B parameter model optimized for inference
   - Full FP16 precision on GPU
   - ~14GB VRAM usage

2. DeepSeek-R1-Distill-Qwen-14B (GGUF)
   - 14B parameter model with 4-bit quantization
   - Reduced memory footprint
   - ~8GB VRAM usage

## Technical Requirements
- NVIDIA GPU with CUDA support
- 16GB+ VRAM recommended
- Python 3.8+
- PyTorch with CUDA
- FastAPI for API endpoints
- Transformers library
- llama-cpp-python for GGUF support

## API Endpoints

### POST /infer
Request body:
```json
{
  "prompt": "Hello, how are you?"
}
```

## Performance Metrics
- Average inference time: 2-4 seconds for 100 tokens
- Memory efficiency: Optimized for consumer GPUs
- Response coherence: Enhanced with post-processing
- Token generation speed: ~25-30 tokens/second

## Future Improvements
- [ ] Flash Attention 2 support
- [ ] Multi-GPU support
- [ ] Streaming responses
- [ ] Model quantization options
- [ ] Response caching

## Development Notes
This project demonstrates expertise in:
- GPU optimization for ML inference
- API design and implementation
- Model loading and memory management
- Error handling and logging
- Performance monitoring and metrics

By Adrian Vargas