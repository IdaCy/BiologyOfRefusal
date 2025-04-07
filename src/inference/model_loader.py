import os
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Gemma3ForConditionalGeneration

def install_and_load_model(model_name, hf_token, max_seq_length, logger=None):
    """
    If the user picks 'gemma-3': custom hack. 
    Otherwise, normal HF load.
    Returns: (tokenizer, model)

    Loads models:
      - Gemma-3 => special gemma-3 class
      - Gemma-2 => normal HF class, placed entirely on GPU
      - Gwen => normal HF fallback
      - GPT-Neo, Pythia, OPT, Cerebras => each placed entirely on GPU
      - else => generic fallback
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "gemma-3" in model_name.lower():
        # Optionally- custom pip install:
        # subprocess.check_call(["pip", "install", "-q", "git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3"])
        # os._exit(0)  # if absolutely required, but that kills the Python process...
        
        # Then load the Gemma3 model
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            token=hf_token,
            torch_dtype=torch.float32,
            max_memory={0: "20GB", "cpu": "30GB"} 
        )
        model.to(device)
        if logger:
            logger.info("Loaded Gemma-3 model.")
        
    # ----------------------------------------------------------------
    # Gemma-2
    # ----------------------------------------------------------------
    elif "gemma-2" in model_name.lower():
        # Gemma-2 => load entire model on GPU or CPU
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.bfloat16
        )
        model.to(device)
        if logger:
            logger.info(f"Loaded Gemma-2 model on {device}.")

    # ----------------------------------------------------------------
    # GPT-Neo 1.3B
    # ----------------------------------------------------------------
    elif "gpt-neo-1.3b" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16
        )
        model.to(device)
        if logger:
            logger.info(f"Loaded GPT-Neo 1.3B on {device}.")

    # ----------------------------------------------------------------
    # Pythia 1.4B
    # ----------------------------------------------------------------
    elif "pythia-1.4b" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16
        )
        model.to(device)
        if logger:
            logger.info(f"Loaded Pythia-1.4B on {device}.")

    # ----------------------------------------------------------------
    # OPT-1.3b
    # ----------------------------------------------------------------
    elif "opt-1.3b" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16
        )
        model.to(device)
        if logger:
            logger.info(f"Loaded OPT-1.3B on {device}.")

    # ----------------------------------------------------------------
    # Cerebras-GPT-1.3B
    # ----------------------------------------------------------------
    elif "cerebras-gpt-1.3b" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16
        )
        model.to(device)
        if logger:
            logger.info(f"Loaded Cerebras-GPT-1.3B on {device}.")

    # ----------------------------------------------------------------
    # Gwen
    # ----------------------------------------------------------------
    elif "gwen" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer.padding_side = "left"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token
        )
        model.to(device)
        if logger:
            logger.info(f"Loaded Gwen model on {device}.")

    # ----------------------------------------------------------------
    # Fallback (generic HF model)
    # ----------------------------------------------------------------
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        model.to(device)
        if logger:
            logger.info(f"Loaded generic model on {device}.")

    return tokenizer, model
