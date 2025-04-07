def get_config_for(model_name, dataset, hf_token):
    """
    Return a config dictionary.
    """
    # Chop of the short name bit for output naming
    model_short = model_name.split("/")[-1]

    # Chop of the dataset name for output naming
    # Split by '/' to get the last part, then split by '.' and take the first element.
    dataset_name = dataset.split("/")[-1].split(".")[0]

    config = {
        "prompt_file": "data/full_levels.json",
        "output_dir": "outputs/" + model_short + "/" + dataset_name,
        "model_name": model_name,
        "dataset": dataset,
        "hf_token": hf_token,
        "batch_size": 64,
        "max_seq_length": 2048,
        "use_bfloat16": False,
        "log_file": "logs/jb_run" + model_short + dataset_name + "_progress.log",
        "error_log": "logs/jb_run" + model_short + dataset_name + "_error.log",
    }
    return config
