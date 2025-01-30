from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://143.198.37.9:5173"],  # Add frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Model paths configuration
MODEL_PATHS = {
    "7b": "/home/enkisys/leonardogpu/models/deepseek-r1-distill-qwen-7b",
    "1.5b": "/home/enkisys/leonardogpu/models/deepseek-r1-distill-qwen-1.5b"
}

# System prompt configuration
SYSTEM_PROMPT = """You are a knowledgeable and precise assistant"""

# Request model
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.6  # Default to recommended value
    top_p: float = 0.95       # Default to recommended value
    top_k: int = 50           # Default to recommended value
    model_version: str = "7b"
    use_chain_of_thought: bool = True  # Enable/disable CoT

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

# Clean and format the response
def format_response(response):
    # Remove incomplete sentences
    if not any(response.endswith(p) for p in [".", "!", "?", ":", ";", ")"]):
        response = response.rsplit(".", 1)[0] + "." if "." in response else response
    
    # Remove redundant prefixes
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    return response

def is_quality_response(response):
    """
    Check if the response meets quality standards.
    Returns True if the response is coherent, False otherwise.
    """
    # List of patterns that indicate a low-quality response
    nonsense_patterns = [
        "Wait no", "I think", "Maybe", "Perhaps", "Not sure",
        "Wait—", "Wait,", "Wait.", "Wait!", "Wait?"
    ]
    
    # If any nonsense pattern is found, return False
    return not any(pattern in response for pattern in nonsense_patterns)

@app.post("/infer")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received request: {request}")
        
        # Modify the prompt
        full_prompt = f"""{SYSTEM_PROMPT}

User: {request.prompt}

Assistant: <think>
Let me analyze this step by step:
1. Understand the question
2. Recall relevant facts
3. Structure the answer
</think>
<answer>"""
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        
        start_inference_time = time.time()
        
        # Optimized generation parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=0.6,  # Recommended range: 0.5-0.7
            top_p=0.95,       # Recommended for better diversity
            top_k=50,         # Limit vocabulary choices
            repetition_penalty=1.2,  # Prevent repetition
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
            do_sample=True,   # Enable sampling for diverse responses
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            length_penalty=1.0  # Balanced length control
        )
        
        inference_time = time.time() - start_inference_time
        
        # Parse response into thinking process and answer
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = format_response(response)
        
        # Log metrics
        tokens_used = outputs.shape[1]
        tokens_per_second = tokens_used / inference_time
        logger.info(f"Thinking process: {response}")
        
        if not is_quality_response(response):
            response = "I'm sorry, I couldn't generate a proper response. Please try again."
        
        # Update the metrics section
        metrics = {
            "response_quality": is_quality_response(response),
            "response_length": len(response.split())
            # Removed calculate_coherence for now
        }

        logger.info(f"Response metrics: {metrics}")
        
        return {
            "thinking": response,
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