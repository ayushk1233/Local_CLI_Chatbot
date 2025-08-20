import argparse
import sys
import gc
import torch
import os

# Set environment variables to prevent multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from model_loader import load_text_generation_pipeline, FallbackChatbot
from chat_memory import ChatMemory

BANNER = """\
Local CLI Chatbot (Hugging Face)
Type your messages below. Commands: /exit, /reset, /help
"""

HELP = """\
Commands:
  /exit   - quit the chatbot
  /reset  - clear the conversation memory
  /help   - show this help
Options at launch:
  --model_name           (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  --window_size          turns of memory to retain (default: 4)
  --max_new_tokens       (default: 64 for TinyLlama, 80 for Qwen3)
  --temperature          (default: 0.7; lower = more deterministic)
  --top_p                (default: 0.9)
  --greedy               (flag; use greedy decoding instead of sampling)
  --device               (auto|cpu|cuda|mps) - force device selection
"""

def parse_args():
    p = argparse.ArgumentParser(description="Local CLI Chatbot using Hugging Face pipelines")
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF model id for text-generation")
    p.add_argument("--window_size", type=int, default=4, help="Number of previous turns to keep")
    p.add_argument("--max_new_tokens", type=int, default=64, help="Tokens to generate per reply")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling p")
    p.add_argument("--greedy", action="store_true", help="Use greedy decoding (ignore sampling params)")
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Device to use for inference")
    return p.parse_args()

def safe_generate(generator, prompt, gen_kwargs, tokenizer):
    """Safely generate text with error handling and memory management."""
    try:
        # Clear memory before generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Ensure pad/eos tokens are set
        if "pad_token_id" not in gen_kwargs:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        if "eos_token_id" not in gen_kwargs:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

        # Generate with M1-optimized settings
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = generator(prompt, **gen_kwargs)
        
        raw_bot = outputs[0]["generated_text"] if outputs else ""
        
        # Extract only the generated part (remove the input prompt)
        if raw_bot.startswith(prompt):
            raw_bot = raw_bot[len(prompt):].strip()
        
        # Post-process: cut at the next "User:" or "Assistant:" if model keeps rambling
        cut_points = ["\nUser:", "\nAssistant:", "User:", "Assistant:", "\n\n", "\n"]
        for cp in cut_points:
            idx = raw_bot.find(cp)
            if idx != -1:
                raw_bot = raw_bot[:idx]
        
        bot_reply = raw_bot.strip()

        # Fallback if empty generation
        if not bot_reply:
            bot_reply = "I'm sorry, I couldn't generate a proper response. Could you please rephrase your question?"
            
        return bot_reply
        
    except Exception as e:
        print(f"Generation error: {e}")
        return "I apologize, but I encountered an error while generating a response. Please try again."

def main():
    args = parse_args()
    print(BANNER)

    try:
        # Load pipeline
        generator = load_text_generation_pipeline(
            model_name=args.model_name,
            device_preference=args.device,
            default_gen_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "do_sample": (not args.greedy),
                "temperature": args.temperature if not args.greedy else None,
                "top_p": args.top_p if not args.greedy else None,
                "repetition_penalty": 1.1,
                "return_full_text": False,
                "no_repeat_ngram_size": 2,
                "use_cache": True,
            },
        )
        
        # Check if we're using fallback mode
        using_fallback = generator is None
        if using_fallback:
            print("Using fallback chatbot system (no AI model loaded)")
            fallback_bot = FallbackChatbot()
            tokenizer = None
        else:
            tokenizer = generator.tokenizer

        memory = ChatMemory(window_size=args.window_size)

        while True:
            try:
                user_text = input("User: ").strip()
            except EOFError:
                # e.g., Ctrl-D on Unix
                print("\nExiting chatbot. Goodbye!")
                break

            if not user_text:
                continue

            if user_text.lower() == "/exit":
                print("Exiting chatbot. Goodbye!")
                break
            if user_text.lower() == "/reset":
                memory.reset()
                print("(memory cleared)")
                continue
            if user_text.lower() == "/help":
                print(HELP)
                continue

            # Generate response based on mode
            if using_fallback:
                # Use fallback system
                bot_reply = fallback_bot.get_response(user_text)
            else:
                # Build prompt with recent history + new user message
                prompt = memory.build_prompt(user_text)

                # Prepare generation kwargs
                gen_kwargs = generator.default_gen_kwargs.copy()

                # Generate response safely
                bot_reply = safe_generate(generator, prompt, gen_kwargs, tokenizer)

            print(f"Bot: {bot_reply}")

            # Update memory
            memory.add_user(user_text)
            memory.add_bot(bot_reply)

    except KeyboardInterrupt:
        print("\nExiting chatbot. Goodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        print("The chatbot encountered a critical error and must exit.")
    finally:
        # Clean up resources
        try:
            if 'generator' in locals() and generator is not None:
                del generator
            if 'tokenizer' in locals() and tokenizer is not None:
                del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

if __name__ == "__main__":
    main()
