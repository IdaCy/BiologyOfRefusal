import os
import torch

def manage_gpu_memory(logger=None):
    """
    Clears CUDA cache, sets environment variables for memory usage,
    and returns a max_memory dict for device_map usage.
    """
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        max_mem_str = f"{int(gpu_mem * 0.9)}GB"
        if logger:
            logger.info(f"Setting max GPU memory to {max_mem_str}")
        max_memory = {0: max_mem_str}
        return max_memory
    else:
        raise RuntimeError("No GPUs available!")
