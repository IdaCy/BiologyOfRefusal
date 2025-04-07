# BiologyOfRefusal

## Overview

This repository contains code to load and analyze language model activations (large models like Gemma-3). It performs:

- Inference on harmful vs. benign prompts, capturing hidden states for downstream interpretability.

- Directions-based analysis to see how certain vectors (e.g., “harmful vs. harmless,” “refused vs. not refused”) appear in the model’s hidden state space.

- Visualization of how projections differ across layers and across different categories of prompts.

In addition, there is a steering component, where an additional notebook (steering.ipynb) shows how to manipulate or guide the model’s output using identified directions or gating strategies.

## Structure

```
BiologyOfRefusal/
├── environment.yml
├── README.md
├── logs/
├── notebooks/                      # Main logic happens here
│   ├── main_analysis.ipynb
│   └── steering.ipynb
├── src/                            # Imports to notebooks
│   ├── config/
│   │   └── config_manager.py
│   ├── inference/
│   │   ├── model_loader.py
│   │   ├── capture_activations.py
│   │   └── memory_manager.py
│   ├── visualization/
│   │   └── plot_utils.py
│   └── utils/
│       ├── logging_config.py       # Global logger
│       ├── text_filters.py         # Check if output is refused
│       └── direction_utils.py
└── .gitignore
```

## Folders and Files:

environment.yml: Defines the Conda environment for installing dependencies (PyTorch, Transformers, matplotlib, etc.).

notebooks/:

- main_analysis.ipynb: The main notebook that runs inference, collects hidden states, classifies output as refused or not, and does interpretability analysis.

- steering.ipynb: An additional notebook demonstrating how to “steer” or manipulate the model’s behavior using the discovered directions or gating mechanisms.

src/: Contains all Python modules that the notebooks import:

config/: Centralized configuration logic.

inference/: GPU memory management, model loading (including special Gemma-3 logic), capturing activations.

utils/: Logging setup, text filtering (for refusal keywords), direction math (difference vectors, projections, etc.).

visualization/: Helper functions for plotting histograms, line charts, etc.

data/: A JSON file containing benign prompts (e.g., benign_prompts.json), so they can be read in the main notebook.

## Setup and Installation

### Clone the Repository
```
git clone git@github.com:IdaCy/BiologyOfRefusal.git
cd my_project
```

### Create and Activate Conda Environment

```
conda env create -f environment.yml
conda activate harmbenignrefusal
```

### Running the Main Notebook

### Start Jupyter (/ preferred environment)
```
jupyter notebook
```

### Open notebooks/main_analysis.ipynb and run all cells.

- You can edit the top-level variables like model_name and dataset to switch which model or data is used.

- The notebook will output logs to logs/run.log by default, and produce plots inline.

### Steering Notebook

- steering.ipynb is a second notebook demonstrating how to manipulate or guide the model outputs, using the directions or gating logic discovered in main_analysis.ipynb.

## Notes

- GPU Usage: The code assumes CUDA is available. If you need to run on CPU, the memory_manager.py logic and model_loader.py logic should be adapted accordingly.

- If using Gemma-3 (or other custom models), the model_loader.py may do special installations. For standard models, the code uses normal Hugging Face behaviors.

## Troubleshooting

- Ensure the number of “harmful” vs. “benign” prompts matches your indexing. The current code expects 120 harmful prompts plus benign prompts in benign_prompts.json.

- If GPU memory is insufficient, adjust BATCH_SIZE or reduce the fraction of GPU memory used in memory_manager.py.

## Feel free to PR!
