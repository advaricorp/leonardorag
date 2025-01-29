from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Model paths configuration
MODEL_PATHS = {
    "deepseek14b": "/home/enkisys/leonardogpu/models/DeepSeek-R1-Distill-Qwen-14B-4bit.gguf",
    "mistral7b": "/home/enkisys/leonardogpu/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
}

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.3
    model_version: str = "deepseek14b"  # Default to DeepSeek 14B

# Load model at startup
def load_model(model_version):
    model_path = MODEL_PATHS.get(model_version)
    if not model_path:
        raise ValueError(f"Invalid model version: {model_version}")
    
    logger.info(f"Loading model from: {model_path}")
    start_load_time = time.time()
    
    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,  # Use all GPU layers
        n_ctx=4096,       # Context window
        n_batch=512,      # Batch size for prompt processing
        verbose=True      # Enable verbose logging
    )
    
    load_time = time.time() - start_load_time
    logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
    return model

# Initialize default model (DeepSeek 14B)
default_model_version = "deepseek14b"
model = load_model(default_model_version)

# Warm-up inference
logger.info("Performing warm-up inference...")
warm_up_output = model("Warm-up", max_tokens=1)
logger.info(f"Warm-up response: {warm_up_output}")

@app.post("/infer")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received request: {request}")
        
        start_inference_time = time.time()
        
        # Generate response with optimized parameters
        output = model(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=0.7,
            top_k=50,
            repeat_penalty=1.2,
            stop=["</s>", "\n\n"],  # Stop tokens
            echo=False              # Don't include prompt in response
        )
        
        inference_time = time.time() - start_inference_time
        
        # Extract response and clean it up
        response = output["choices"][0]["text"].strip()
        if not any(response.endswith(p) for p in [".", "!", "?", ":", ";", ")"]):
            response = response.rsplit(".", 1)[0] + "." if "." in response else response
        
        # Calculate metrics
        tokens_used = len(model.tokenize(response.encode()))
        tokens_per_second = tokens_used / inference_time
        
        # Log metrics
        logger.info(f"Generated response: {response}")
        logger.info(f"Inference time: {inference_time:.2f} seconds")
        logger.info(f"Tokens used: {tokens_used}")
        logger.info(f"Tokens per second: {tokens_per_second:.2f}")
        
        return {
            "response": response,
            "inference_time": f"{inference_time:.2f} seconds",
            "tokens_used": tokens_used,
            "tokens_per_second": f"{tokens_per_second:.2f}"
        }
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Note: Different port from safetensors