import torch

def capture_activations(
    text_batch,
    tokenizer,
    model,
    max_seq_length=2048,
    batch_idx=None,
    logger=None
):
    """
    Runs model inference on a batch and extracts required activations.
    Returns hidden states for both input sequence and generated tokens separately.
    Uses deterministic generation (temperature=0).
    """

    try:
        # 1. Tokenize
        encodings = tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        # 2. Generate new tokens with deterministic settings
        with torch.no_grad():
            generated_sequences = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                num_beams=1,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        # 3. Aggregate hidden states
        if hasattr(generated_sequences, 'hidden_states'):
            hidden_states = [torch.stack(layer) for layer in generated_sequences.hidden_states]
            hidden_states = torch.cat(hidden_states, axis=2)
        else:
            hidden_states = None

        # 4. Decode
        final_predictions = [
            tokenizer.decode(seq, skip_special_tokens=True)
            for seq in generated_sequences.sequences.cpu()
        ]

        return {
            "hidden_states": hidden_states,
            "input_ids": input_ids.cpu(),
            "generated_sequences": generated_sequences.sequences.cpu(),
            "final_predictions": final_predictions,
        }

    except Exception as e:
        if logger:
            logger.error(f"Error processing batch {batch_idx}: {e}", exc_info=True)
        else:
            print(f"Error processing batch {batch_idx}: {e}")
        return None
