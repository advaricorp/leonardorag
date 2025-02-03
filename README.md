# Leonardo: Personal AI Assistant with Document Knowledge

**By Adrian Vargas**

## Project Overview

Leonardo is a personal AI assistant designed to act as your "second brain," capable of understanding and responding to your queries using both its general knowledge and your personal documents. This project integrates advanced features like Retrieval-Augmented Generation (RAG) to provide context-aware and personalized assistance.

## Technical Architecture

Leonardo utilizes a modular, hybrid architecture for optimal performance and scalability:

1.  **Web Frontend (React):** A user-friendly web interface for text and voice interaction, built with React and Vite for rapid development and responsiveness.

2.  **Leonardo Core Backend (FastAPI):** The central API built with FastAPI in Python, orchestrating the key functionalities:
    *   **LLM Inference:**  Leverages a local, high-performance DeepSeek-R1-Distill-Qwen-7B model running on a dedicated GPU server for fast and efficient response generation.
    *   **Retrieval-Augmented Generation (RAG):** Integrates ChromaDB Vector Database to index and retrieve information from user-uploaded documents, enhancing the LLM's responses with personalized knowledge.
    *   **Voice Interaction:** Connects to ElevenLabs for high-quality Text-to-Speech (TTS) and Whisper API for accurate Speech-to-Text (STT) capabilities.

3.  **Vector Database (ChromaDB):**  Employed for storing and efficiently retrieving vector embeddings of document chunks, enabling semantic search and RAG.

4.  **LLM Inference Server (FastAPI Endpoint):** A dedicated FastAPI endpoint optimized for running the DeepSeek-R1-Distill-Qwen-7B model on a local NVIDIA RTX 4090 GPU, ensuring low latency and high throughput.

## Key Features

*   **Text-to-Text Interaction:**  Engage in natural language conversations with Leonardo through the web frontend, receiving text-based responses powered by the local DeepSeek LLM.
*   **Voice Interaction (Basic):**  Basic voice input (Speech-to-Text via Whisper API) and voice output (Text-to-Speech via ElevenLabs API) are integrated for a more natural conversational experience.
*   **Retrieval-Augmented Generation (RAG):**  Upload your personal documents, and Leonardo will intelligently search and retrieve relevant information from them to provide contextually enriched and personalized answers.
*   **Document Uploads:**  Upload and index various document types (initially `.txt` files, expandable to PDFs, DOCs) into Leonardo's knowledge base.
*   **High-Performance Local LLM Inference:**  Utilizes a local DeepSeek-R1-Distill-Qwen-7B model running on a dedicated GPU (RTX 4090) for fast and efficient inference, ensuring quick response times.
*   **Modular and Scalable Architecture:**  Dockerized components for easy deployment, portability, and scalability, allowing for future expansion and feature additions.
*   **Basic Authentication:**  Token-based authentication for securing access to the Leonardo Core backend.
*   **Advanced Frontend UI:**  React-based frontend with features like:
    *   Chat History: Maintains a record of past conversations for context and continuity.
    *   Typing Effect: Simulates real-time interaction, enhancing user experience.
    *   Metrics Display: Provides insights into backend performance (inference time, token usage).
    *   Adjustable Model Parameters: Allows fine-tuning of LLM behavior via temperature, max tokens, and other sampling parameters.
*   **Error Handling and Logging:** Implemented throughout the system for robust performance and easier debugging.

## Performance Optimizations

*   **GPU Acceleration:** Leverages NVIDIA RTX 4090 GPU for accelerated LLM inference and embedding generation.
*   **FP16 Precision:** Utilizes FP16 (mixed-precision) for faster and memory-efficient GPU inference.
*   **Flash Attention 2:** Enabled for optimized attention mechanisms in the LLM.
*   **Quantization:** Employs 4-bit quantization for the DeepSeek-R1-Distill-Qwen-14B model (GGUF endpoint) to reduce memory footprint.
*   **Batch Processing:** Optimized batch size for efficient GPU utilization.
*   **Warm-up Inference:** Preloads the model and performs a warm-up inference at server startup to minimize first-request latency.
*   **Context Window Optimization:**  Context length is adjusted to balance performance and information processing capacity.

## Models Supported

*   **DeepSeek-R1-Distill-Qwen-7B (SafeTensors Endpoint - Port 8000):**
    *   7B parameter model optimized for high-performance inference.
    *   Full FP16 precision on GPU.
    *   ~14GB VRAM usage.

*   **DeepSeek-R1-Distill-Qwen-14B (GGUF Endpoint - Port 8001):**
    *   14B parameter model with 4-bit quantization for memory efficiency.
    *   Reduced memory footprint, ~8GB VRAM usage.
    *   Excellent reasoning capabilities for its size.

## Technical Requirements

*   NVIDIA GPU with CUDA support (16GB+ VRAM recommended, RTX 4090 for optimal performance)
*   Python 3.8+
*   PyTorch with CUDA Toolkit
*   FastAPI, Uvicorn
*   Transformers library, Accelerate, Flash Attention 2
*   `llama-cpp-python`
*   ChromaDB Python Client
*   `sentence-transformers` library
*   ElevenLabs API Key
*   OpenAI API Key (for Whisper)
*   Docker, Docker Compose

## API Endpoints

*   **POST `/infer`**:
    *   Handles text-based queries and returns text responses from the LLM.
    *   Request body (JSON):
        ```json
        {
          "prompt": "Hello, how are you?",
          "model_choice": 1  // Optional: 1 for Mistral 7B, 2 for DeepSeek-R1-Distill-Qwen-14B (GGUF Endpoint only)
        }
        ```
    *   Response body (JSON):
        ```json
        {
          "response": "...",
          "inference_time": "...",
          "tokens_used": ...,
          "tokens_per_second": "..."
        }
        ```
*   **POST `/upload_doc`**:
    *   Handles document uploads for indexing in the Vector Database.
    *   Request body (multipart/form-data):
        *   `file`:  The document file to upload.
        *   `token`: Authentication token.
    *   Response body (JSON):
        ```json
        {
          "message": "Document uploaded and indexed successfully"
        }
        ```
*   **POST `/ask`**:
    *   Handles text-based queries to Leonardo Core, leveraging Retrieval-Augmented Generation (RAG).
    *   Request body (JSON):
        ```json
        {
          "text": "Question for Leonardo",
          "token": "YOUR_TOKEN_SECRETO"
        }
        ```
    *   Response body (JSON):
        ```json
        {
          "response": "...", // Text response from Leonardo, informed by RAG
          "audio_response_url": "...", // URL to the generated TTS audio (future)
          "inference_time": "...",
          "tokens_used": "..."
        }
        ```
*   **POST `/stt`**:
    *   Handles audio uploads for Speech-to-Text transcription (future).
    *   Request body (multipart/form-data or JSON with audio URL):
        *   `audio`: Audio file (or URL).
        *   `token`: Authentication token.
    *   Response body (JSON):
        ```json
        {
          "transcription": "..."
        }
        ```

## Performance Metrics

*   **Average inference time:** 2-4 seconds for 100 tokens (DeepSeek-R1-Distill-Qwen-7B). Under 1 second for Mistral 7B.
*   **Token generation speed:** ~25-30 tokens/second (DeepSeek-R1-Distill-Qwen-7B). Significantly faster for Mistral 7B.
*   **Memory efficiency:** Optimized for consumer GPUs, with quantized models available for reduced VRAM usage.
*   **Response coherence:** Enhanced with post-processing and RAG context integration.

## Future Improvements

*   Advanced RAG Techniques: Implement more sophisticated chunking, retrieval, and prompt augmentation strategies.
*   Multi-Document Support: Enhance document handling to manage and search across multiple uploaded documents effectively.
*   Persistent Memory: Integrate persistent storage for chat history and user data using a database (e.g., PostgreSQL, Redis).
*   AI Agent Framework: Develop a robust agent orchestration layer using frameworks like LangChain to enable more complex task automation and tool usage.
*   More API Tools: Integrate with external APIs (Google Calendar, Home Assistant, etc.) to expand Leonardo's capabilities.
*   Improved Voice Interaction: Enhance voice input and output with streaming responses and more natural voice control.
*   Enhanced Security: Implement robust authentication and authorization mechanisms (e.g., JWT, OAuth2).
*   Scalability and Monitoring: Migrate to Kubernetes for autoscaling and implement comprehensive monitoring and logging.
*   Frontend UI/UX Enhancements: Develop a more visually appealing and feature-rich user interface with dashboards, visualizations, and user settings.
*   Raspberry Pi 5 Integration:  Optimize and deploy components to run efficiently on a Raspberry Pi 5 for local processing and wake-word detection.

## Development Notes

This project showcases expertise in:

*   Building a functional and high-performance AI assistant from scratch.
*   Integrating and optimizing Large Language Models for local GPU inference.
*   Implementing Retrieval-Augmented Generation (RAG) for knowledge enhancement.
*   Designing and developing RESTful APIs with FastAPI.
*   Creating a responsive user interface with React.
*   Dockerizing and deploying complex applications.
*   Optimizing for performance, memory efficiency, and user experience in AI applications.

**By Adrian Vargas**
