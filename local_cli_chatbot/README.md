# Local CLI Chatbot

A robust, local command-line chatbot powered by Hugging Face's language models. This chatbot maintains conversation history, provides intelligent responses, and runs entirely on your local machine without requiring internet connectivity after initial model download.

## Features

- **Local Operation**: Runs completely offline after model download
- **Conversation Memory**: Maintains context using a sliding window buffer
- **Multiple Model Support**: Automatically falls back to compatible models
- **GPU Acceleration**: Supports Apple Silicon (MPS) and CUDA for faster inference
- **M1 Optimized**: Specifically tuned for Apple Silicon performance
- **Robust Error Handling**: Gracefully handles generation failures and memory issues
- **Memory Efficient**: Optimized for both CPU and GPU usage with fallback models
- **Clean CLI Interface**: Simple commands and graceful exit handling

## Requirements

- Python 3.8+
- 4GB+ RAM (for medium models)
- 8GB+ RAM (for GPU showcase models)
- Internet connection (for initial model download only)
- **For GPU users**: PyTorch with CUDA or Apple Silicon support

## Installation

1. **Clone or download the project files**
2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **For GPU users (optional but recommended)**:
   ```bash
   # For CUDA users
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For Apple Silicon users
   pip install torch torchvision torchaudio
   ```

## Usage

### Basic Usage

Start the chatbot with default settings:
```bash
source venv/bin/activate
python interface.py
```

### GPU Showcase Models (Recommended for Apple Silicon / GPU users)

For users with Apple Silicon Macs or CUDA GPUs, we recommend using these models:

```bash
# TinyLlama-1.1B-Chat (default, fastest on M1)
python interface.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Qwen3-1.7B-ShiningValiant3 (alternative, smarter and still fast on M1)
python interface.py --model_name ValiantLabs/Qwen3-1.7B-ShiningValiant3
```

### CLI Commands for All Available Models

#### **GPU Models (Apple Silicon / CUDA)**

**1. TinyLlama (Default - Fastest on M1)**
```bash
python interface.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device mps
```

**2. Qwen3 (Better Quality)**
```bash
python interface.py --model_name ValiantLabs/Qwen3-1.7B-ShiningValiant3 --device mps
```

**3. DialoGPT Medium (Alternative)**
```bash
python interface.py --model_name microsoft/DialoGPT-medium --device mps
```

**4. GPT-2 (Standard)**
```bash
python interface.py --model_name gpt2 --device mps
```

#### **CPU Fallback Models**

**5. DialoGPT Medium (CPU Alternative)**
```bash
python interface.py --model_name microsoft/DialoGPT-medium --device cpu
```

**6. DialoGPT Small (CPU Fallback)**
```bash
python interface.py --model_name microsoft/DialoGPT-small --device cpu
```

### Step-by-Step Model Testing

1. **Navigate to project directory:**
   ```bash
   cd /Users/ayuwat/Desktop/local_cli_chatbot
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Test each model one by one:**

   **Start with TinyLlama (recommended):**
   ```bash
   python interface.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device mps
   ```
   
   **After testing, exit with `/exit`, then test Qwen3:**
   ```bash
   python interface.py --model_name ValiantLabs/Qwen3-1.7B-ShiningValiant3 --device mps
   ```
   
   **Continue with other models...**

### Quick Test Commands

**Test all GPU models in sequence:**
```bash
# 1. TinyLlama
python interface.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device mps

# 2. Qwen3  
python interface.py --model_name ValiantLabs/Qwen3-1.7B-ShiningValiant3 --device mps

# 3. DialoGPT-medium
python interface.py --model_name microsoft/DialoGPT-medium --device mps

# 4. GPT-2
python interface.py --model_name gpt2 --device mps
```

**Test CPU fallback models:**
```bash
# 5. DialoGPT-medium
python interface.py --model_name microsoft/DialoGPT-medium --device cpu

# 6. DialoGPT-small
python interface.py --model_name microsoft/DialoGPT-small --device cpu
```

### Advanced Options

```bash
python interface.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device mps \
  --window_size 5 \
  --max_new_tokens 64 \
  --temperature 0.7 \
  --top_p 0.9
```

### Command Line Arguments

- `--model_name`: Hugging Face model ID (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- `--device`: Device selection (`auto|cpu|cuda|mps`)
- `--window_size`: Number of conversation turns to remember (default: 4)
- `--max_new_tokens`: Maximum tokens to generate per response (default: 64 for TinyLlama, 80 for Qwen3)
- `--temperature`: Sampling temperature, 0.0-1.0 (default: 0.7)
- `--top_p`: Nucleus sampling parameter (default: 0.9)
- `--greedy`: Use deterministic generation instead of sampling

### Chat Commands

- `/exit` - Exit the chatbot
- `/reset` - Clear conversation memory
- `/help` - Show help information

## Sample Interaction

```
Local CLI Chatbot (Hugging Face)
Type your messages below. Commands: /exit, /reset, /help

Device set to use GPU (Apple Silicon)
Attempting to load model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Successfully loaded requested model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

User: What is the capital of France?
Bot: The capital of France is Paris.

User: And what about Italy?
Bot: The capital of Italy is Rome.

User: Tell me about both cities
Bot: Paris, the capital of France, is known as the "City of Light" and is famous for its art, fashion, gastronomy, and culture. It's home to iconic landmarks like the Eiffel Tower and Louvre Museum.

Rome, the capital of Italy, is the "Eternal City" with a rich history spanning over 2,500 years. It's famous for its ancient ruins like the Colosseum and Roman Forum, as well as being the center of the Catholic Church with Vatican City.

User: /exit
Exiting chatbot. Goodbye!
```

## Model Information

### GPU Showcase Models (Recommended)

#### `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Default)
- **Size**: ~1.1GB
- **Quality**: Excellent conversational abilities
- **Memory**: ~2-3GB RAM usage
- **Speed**: **Fastest on M1** - optimized for chat
- **Best for**: Apple Silicon Macs, quick responses
- **Characteristics**: Short, coherent, factual responses

#### `ValiantLabs/Qwen3-1.7B-ShiningValiant3`
- **Size**: ~1.7GB
- **Quality**: **High-quality conversational model**
- **Memory**: ~3-4GB RAM usage
- **Speed**: **Great balance** - smarter than TinyLlama, still fast on M1
- **Best for**: Users who want better quality while maintaining speed
- **Characteristics**: Better quality responses, optimized for M1 with MPS

### CPU Fallback Models
If GPU models fail to load, the system automatically tries:
1. `microsoft/DialoGPT-medium` (~774MB)
2. `gpt2` (~548MB)
3. `microsoft/DialoGPT-small` (~117MB)

### Complete Model Comparison Table

| Model | Size | Speed | Quality | Best For | Device |
|-------|------|-------|---------|----------|---------|
| **TinyLlama** | ~1.1GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Fastest responses | MPS/CUDA |
| **Qwen3** | ~1.7GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Best quality | MPS/CUDA |
| **DialoGPT-medium** | ~774MB | ⚡⚡ | ⭐⭐⭐⭐ | Good balance | MPS/CUDA |
| **GPT-2** | ~548MB | ⚡ | ⭐⭐⭐ | Standard | MPS/CUDA |
| **DialoGPT-medium** | ~774MB | ⚡⚡ | ⭐⭐⭐⭐ | CPU alternative | CPU |
| **DialoGPT-small** | ~117MB | ⚡ | ⭐⭐ | Lightweight | CPU |

### Expected Performance by Model

- **TinyLlama**: ~2-3 seconds (fastest)
- **Qwen3**: ~3-4 seconds (best quality)
- **DialoGPT-medium**: ~3-4 seconds
- **GPT-2**: ~4-5 seconds
- **CPU models**: ~5-8 seconds (slower but stable)

### Model Selection Guide

**For Speed (M1 Users):**
- **Best**: TinyLlama - fastest responses
- **Alternative**: DialoGPT-medium - good CPU option

**For Quality:**
- **Best**: Qwen3 - highest quality responses
- **Alternative**: DialoGPT-medium - good balance

**For Stability:**
- **Best**: CPU models - most stable
- **Alternative**: GPT-2 - reliable GPU option

## Device Support

### Apple Silicon (MPS) - **M1 Optimized**
- **Automatic detection**: Set `--device auto` (default)
- **Manual selection**: Set `--device mps`
- **Models**: TinyLlama, Qwen3, DialoGPT-medium
- **Performance**: **Optimized for speed** with float16, KV cache, and M1-specific optimizations

### CUDA GPUs
- **Automatic detection**: Set `--device auto` (default)
- **Manual selection**: Set `--device cuda`
- **Models**: TinyLlama, Qwen3, DialoGPT-medium

### CPU
- **Automatic fallback**: When GPU is not available
- **Manual selection**: Set `--device cpu`
- **Models**: microsoft/DialoGPT-medium, gpt2, DialoGPT-small

## Performance Optimizations

### M1-Specific Optimizations
- **Float16 precision**: Faster inference with minimal quality loss
- **KV Cache**: Enabled for faster token generation
- **Reduced token limits**: 64 tokens for TinyLlama, 80 for Qwen3 for faster responses
- **Flash Attention 2**: When available for faster attention computation
- **Model half-precision**: Automatic conversion for MPS devices

### Speed vs Quality Balance
- **TinyLlama**: Fastest responses, good quality
- **Qwen3**: Better quality, still fast on M1
- **CPU models**: Slower but more stable

## Architecture

### `model_loader.py`
- Handles model and tokenizer loading
- Implements fallback model strategy
- Manages memory and device allocation
- Configures generation parameters for different devices
- **M1-specific optimizations** for faster inference

### `chat_memory.py`
- Maintains conversation history
- Implements sliding window buffer
- Formats context for model input
- Manages conversation flow

### `interface.py`
- Main CLI loop and user interaction
- Handles command processing
- Manages model generation
- Provides error handling and cleanup

## Troubleshooting

### Slow Response Times
- **Use TinyLlama**: Fastest model for M1
- **Reduce max_new_tokens**: Lower values = faster responses
- **Use MPS device**: Ensure `--device mps` for Apple Silicon
- **Close other apps**: Free up memory and GPU resources

### Bus Error / Memory Issues
- The system automatically detects and uses appropriate devices
- Try reducing `--max_new_tokens` or `--window_size`
- Use CPU fallback models with `--device cpu`

### Model Loading Failures
- Check internet connection for initial download
- Ensure sufficient disk space (~2GB for GPU models)
- Verify PyTorch installation and device support

### Poor Response Quality
- **Switch to Qwen3**: Better quality than TinyLlama
- Adjust `--temperature` (lower = more focused)
- Increase `--max_new_tokens` for longer responses
- Use `--greedy` for more deterministic outputs

### GPU Issues
- **Apple Silicon**: Ensure PyTorch supports MPS
- **CUDA**: Verify CUDA toolkit and PyTorch compatibility
- **Memory**: Close other applications to free up VRAM

### Model Testing Issues
- **First run slow**: Models download automatically (~1-7GB depending on model)
- **Subsequent runs**: Instant loading after first download
- **Model switching**: Exit with `/exit` before testing different models
- **Memory errors**: Use smaller models or reduce `--max_new_tokens`
- **Device conflicts**: Ensure `--device mps` for Apple Silicon, `--device cpu` for fallback
- **Known issues**: `distilgpt2` may cause bus errors - removed from fallback models

## Performance Tips

### M1 Users (Recommended)
- **Use TinyLlama**: Fastest responses for chat
- **Use Qwen3**: Better quality when you need it
- **Set device to MPS**: `--device mps` for best performance
- **Keep max_new_tokens low**: 64-80 for fastest responses

### CUDA Users
- **Use Qwen3**: Better quality with GPU acceleration
- **Ensure sufficient VRAM**: 4GB+ recommended
- **Models**: Both models provide excellent GPU performance

### CPU Users
- **Memory**: Close other applications to free up RAM
- **Models**: DialoGPT-medium provides good CPU performance
- **Settings**: Reduce `--max_new_tokens` for faster responses

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the chatbot!

## Quick Reference

### Most Common Commands

**Fastest M1 Performance:**
```bash
python interface.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device mps
```

**Best Quality:**
```bash
python interface.py --model_name ValiantLabs/Qwen3-1.7B-ShiningValiant3 --device mps
```

**CPU Fallback:**
```bash
python interface.py --device cpu
```

**Custom Token Limit:**
```bash
python interface.py --max_new_tokens 30
```

### Chat Commands
- `/exit` - Exit chatbot
- `/reset` - Clear memory
- `/help` - Show help

### Model Download Sizes
- TinyLlama: ~1.1GB
- Qwen3: ~1.7GB
- DialoGPT-medium: ~774MB
- GPT-2: ~548MB
- DistilGPT-2: ~334MB
- DialoGPT-small: ~117MB

## License

This project is open source and available under the MIT License.
