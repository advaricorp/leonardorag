from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Model paths configuration
MODEL_PATHS = {
    "7b": "/home/enkisys/leonardogpu/models/deepseek-r1-distill-qwen-7b",
    "1.5b": "/home/enkisys/leonardogpu/models/deepseek-r1-distill-qwen-1.5b"
}

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.3  # Lower temperature for more focused responses
    model_version: str = "7b"  # Default to 7B model

# Load model at startup
def load_model(model_version):
    model_path = MODEL_PATHS.get(model_version)
    if not model_path:
        raise ValueError(f"Invalid model version: {model_version}")
    
    logger.info(f"Loading model from: {model_path}")
    start_load_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    load_time = time.time() - start_load_time
    logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
    return model, tokenizer

# Initialize default model (7B)
default_model_version = "7b"
model, tokenizer = load_model(default_model_version)

# Warm-up inference
logger.info("Performing warm-up inference...")
warm_up_input = tokenizer("Warm-up", return_tensors="pt").to("cuda")
warm_up_output = model.generate(**warm_up_input, max_new_tokens=1)
warm_up_response = tokenizer.decode(warm_up_output[0], skip_special_tokens=True)
logger.info(f"Warm-up response: {warm_up_response}")

@app.post("/infer")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received request: {request}")
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
        
        start_inference_time = time.time()
        
        # Optimized generation parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            top_p=0.7,  # Lower top_p for more focused sampling
            top_k=50,   # Limit vocabulary choices
            repetition_penalty=1.2,  # Prevent repetition
            no_repeat_ngram_size=3,  # Prevent repeating phrases
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            length_penalty=1.0  # Balanced length control
        )
        
        inference_time = time.time() - start_inference_time
        
        # Clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not any(response.endswith(p) for p in [".", "!", "?", ":", ";", ")"]):
            response = response.rsplit(".", 1)[0] + "." if "." in response else response
        
        # Log metrics
        tokens_used = outputs.shape[1]
        tokens_per_second = tokens_used / inference_time
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
    uvicorn.run(app, host="0.0.0.0", port=8000)