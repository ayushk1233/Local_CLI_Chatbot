from typing import Optional, Dict, Any
import torch
import gc
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class FallbackChatbot:
    """Simple fallback chatbot when model loading fails."""
    
    def __init__(self):
        self.responses = {
            "capital": {
                "france": "The capital of France is Paris.",
                "italy": "The capital of Italy is Rome.",
                "germany": "The capital of Germany is Berlin.",
                "spain": "The capital of Spain is Madrid.",
                "uk": "The capital of the United Kingdom is London.",
                "india": "The capital of India is New Delhi.",
                "japan": "The capital of Japan is Tokyo.",
                "china": "The capital of China is Beijing.",
                "russia": "The capital of Russia is Moscow.",
                "canada": "The capital of Canada is Ottawa.",
                "australia": "The capital of Australia is Canberra.",
                "brazil": "The capital of Brazil is Brasília.",
                "mexico": "The capital of Mexico is Mexico City.",
                "egypt": "The capital of Egypt is Cairo.",
                "south africa": "The capital of South Africa is Pretoria."
            },
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "Greetings! I'm here to assist you.",
                "Hello! I'm ready to help with your questions."
            ],
            "weather": [
                "I'm sorry, I don't have access to real-time weather information.",
                "For current weather, I recommend checking a weather service or app.",
                "I can't provide weather updates, but I can help with other questions!"
            ],
            "help": [
                "I can help you with general knowledge questions, especially about geography, history, and basic facts.",
                "Feel free to ask me about capitals, countries, or other general topics!",
                "I'm here to help with your questions. What would you like to know?"
            ]
        }
    
    def get_response(self, user_input):
        """Generate a response based on user input."""
        user_input = user_input.lower().strip()
        
        # Check for capital questions
        if "capital" in user_input:
            for country, capital in self.responses["capital"].items():
                if country in user_input:
                    return capital
            return "I'm not sure about that country's capital. Could you be more specific?"
        
        # Check for greetings
        if any(word in user_input for word in ["hello", "hi", "hey", "greetings"]):
            import random
            return random.choice(self.responses["greeting"])
        
        # Check for weather questions
        if "weather" in user_input:
            import random
            return random.choice(self.responses["weather"])
        
        # Check for help requests
        if "help" in user_input or "what can you do" in user_input:
            import random
            return random.choice(self.responses["help"])
        
        # Default response
        return "I'm a simple chatbot. I can help with basic questions about capitals, countries, and general knowledge. What would you like to know?"

def load_text_generation_pipeline(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_preference: Optional[str] = "auto",
    default_gen_kwargs: Optional[Dict[str, Any]] = None,
):
    """Create a Hugging Face text-generation pipeline.

    Args:
        model_name: HF model id (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0').
        device_preference: 'auto'|'cpu'|'cuda'|'mps'. If 'auto', uses CUDA if available.
        default_gen_kwargs: default generation params for the pipeline call.

    Returns:
        A transformers pipeline configured for text-generation.
    """
    # Set environment variables to prevent multiprocessing issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Check if we should use GPU (Apple Silicon or CUDA)
    use_gpu = False
    if device_preference == "auto":
        if torch.cuda.is_available():
            use_gpu = True
            device = 0
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            use_gpu = True
            device = "mps"
        else:
            use_gpu = False
            device = -1
    elif device_preference == "cuda" and torch.cuda.is_available():
        use_gpu = True
        device = 0
    elif device_preference == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        use_gpu = True
        device = "mps"
    else:
        use_gpu = False
        device = -1

    print(f"Device set to use {'GPU (Apple Silicon)' if device == 'mps' else 'GPU (CUDA)' if device == 0 else 'CPU'}")

    # Try the requested model first, fallback to other models if it fails
    models_to_try = [model_name]
    
    # Add GPU showcase models if using GPU - optimized for M1 performance
    if use_gpu:
        if model_name != "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
            models_to_try.append("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        if model_name != "ValiantLabs/Qwen3-1.7B-ShiningValiant3":
            models_to_try.append("ValiantLabs/Qwen3-1.7B-ShiningValiant3")
        if model_name != "microsoft/DialoGPT-medium":
            models_to_try.append("microsoft/DialoGPT-medium")
        if model_name != "gpt2":
            models_to_try.append("gpt2")
    else:
        # CPU fallback models - removed distilgpt2 due to stability issues
        if model_name != "gpt2":
            models_to_try.append("gpt2")
        if model_name != "microsoft/DialoGPT-small":
            models_to_try.append("microsoft/DialoGPT-small")
        # Add a more stable small model as alternative
        if model_name != "microsoft/DialoGPT-medium":
            models_to_try.append("microsoft/DialoGPT-medium")
    
    for try_model in models_to_try:
        try:
            print(f"Attempting to load model: {try_model}")
            
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            tokenizer = AutoTokenizer.from_pretrained(try_model)
            
            # Handle special tokens for different model types
            if "tinyllama" in try_model.lower():
                # TinyLlama models need special token handling
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                # Set additional special tokens for better conversation flow
                special_tokens = {
                    'pad_token': tokenizer.eos_token,
                    'additional_special_tokens': ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
                }
                tokenizer.add_special_tokens(special_tokens)
            elif "qwen" in try_model.lower():
                # Qwen models need special token handling
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                special_tokens = {
                    'pad_token': tokenizer.eos_token,
                    'additional_special_tokens': ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
                }
                tokenizer.add_special_tokens(special_tokens)
            elif "dialogpt" in try_model.lower():
                # DialoGPT models need special token handling
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                special_tokens = {
                    'pad_token': tokenizer.eos_token,
                    'additional_special_tokens': ['<|endoftext|>']
                }
                tokenizer.add_special_tokens(special_tokens)
            else:
                # Standard GPT-2 token handling
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token

            # Load model with M1-optimized settings
            if use_gpu and device == "mps":
                # For Apple Silicon, optimize for speed
                model = AutoModelForCausalLM.from_pretrained(
                    try_model,
                    torch_dtype=torch.float16,  # Use float16 for faster inference
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2" if "flash_attention_2" in dir(torch.nn.functional) else None,
                )
                # Move to MPS device manually
                model = model.to("mps")
                # Enable optimizations for faster inference
                model.eval()
                if hasattr(model, 'half'):
                    model = model.half()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    try_model,
                    torch_dtype=torch.float16 if use_gpu else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="auto" if use_gpu and device != "mps" else None,
                )

            # Resize token embeddings if we added special tokens
            if any(x in try_model.lower() for x in ["tinyllama", "qwen", "dialogpt"]):
                model.resize_token_embeddings(len(tokenizer))

            # Create pipeline with M1-optimized settings
            gen = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,  # -1 = CPU, 0 = first CUDA GPU, "mps" = Apple Silicon
                framework="pt",
            )

            # Store defaults on the pipeline object for convenience - optimized for speed
            gen.default_gen_kwargs = default_gen_kwargs or {
                "max_new_tokens": 64 if "tinyllama" in try_model.lower() else 80,  # Shorter for faster responses
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "return_full_text": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "no_repeat_ngram_size": 2,  # Reduced for speed
                "use_cache": True,  # Enable KV cache for faster generation
            }
            
            if try_model != model_name:
                print(f"Successfully loaded fallback model: {try_model}")
            else:
                print(f"Successfully loaded requested model: {try_model}")
                
            return gen
            
        except Exception as e:
            print(f"Failed to load {try_model}: {e}")
            # Clean up on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # If all models fail, return None to use fallback
    print("All models failed to load. Using fallback chatbot system.")
    return None
