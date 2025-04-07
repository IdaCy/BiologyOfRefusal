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
            torch_dtype=torch.float32
        )
        if logger:
            logger.info("Loaded Gemma-3 model.")
        
    elif "gemma-2" in model_name.lower():
        # load the gemma-2 variant 
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            token=hf_token,
            torch_dtype=torch.bfloat16  # Or however Gemma-2 is best loaded
        )
        if logger:
            logger.info("Loaded Gemma-2 model.")

    elif "gwen" in model_name.lower():
        # Example for "Gwen" model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if logger:
            logger.info("Loaded Gwen model.")
    else:
        # fallback: generic model
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if logger:
            logger.info("Loaded generic model.")

    return tokenizer, model
