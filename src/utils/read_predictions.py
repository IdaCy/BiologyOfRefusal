#!/usr/bin/env python
import os
import glob
import torch
import logging
from datetime import datetime
from tqdm import tqdm
import argparse

# Default parameters
DEFAULT_READ_OUTPUT_DIR = "output/extractions/gemma2b/jb"
DEFAULT_MAX_PREDICTIONS = 20
DEFAULT_WRITE_PREDICTIONS_FILE = None
DEFAULT_LOG_FILE = "logs/read_predictions.log"

def read_predictions(
    read_output_dir=DEFAULT_READ_OUTPUT_DIR,
    max_predictions=DEFAULT_MAX_PREDICTIONS,
    write_predictions_file=DEFAULT_WRITE_PREDICTIONS_FILE,
    log_file=DEFAULT_LOG_FILE,
):
    """
    Scans a directory of `.pt` files (named like "activations_XXXX_YYYY.pt"),
    loads them up to a max number of predictions, and optionally writes them
    to a text file.

    Returns a list of all collected predictions (strings).
    """
    # Create logs directory if needed
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # ----------------------------------------------------------------
    # Set up logging
    # ----------------------------------------------------------------
    logger = logging.getLogger("ReadPredictionsLogger")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if logger is already configured
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    logger.info("=== Starting read_predictions function ===")
    logger.info(f"read_output_dir = {read_output_dir}")
    logger.info(f"log_file = {log_file}")
    logger.info(f"max_predictions = {max_predictions}")
    logger.info(f"write_predictions_file = {write_predictions_file}")

    # ----------------------------------------------------------------
    # 2. Locate and sort the .pt files
    # ----------------------------------------------------------------
    pt_files = sorted(glob.glob(os.path.join(read_output_dir, "activations_*.pt")))
    if not pt_files:
        logger.warning(f"No .pt files found in {read_output_dir}. Returning empty list.")
        return []  # Return empty list instead of exiting

    logger.info(f"Found {len(pt_files)} .pt files to process.")

    # ----------------------------------------------------------------
    # 3. Read predictions from each file
    # ----------------------------------------------------------------
    all_predictions = []
    predictions_collected = 0

    for pt_file in tqdm(pt_files, desc="Reading .pt files"):
        logger.debug(f"Loading file: {pt_file}")
        try:
            data = torch.load(pt_file, map_location="cpu")
            # The dictionary includes a "final_predictions" key
            if "final_predictions" in data:
                for pred in data["final_predictions"]:
                    all_predictions.append(pred)
                    predictions_collected += 1
                    if predictions_collected >= max_predictions:
                        logger.info("Reached max_predictions limit; stopping.")
                        break
            else:
                logger.warning(f"No 'final_predictions' key in {pt_file}.")
        except Exception as e:
            logger.exception(f"Could not load {pt_file}: {str(e)}")

        if predictions_collected >= max_predictions:
            break

    logger.info(f"Collected {len(all_predictions)} total predictions.")

    # ----------------------------------------------------------------
    # 4. Optionally print sample + write to file
    # ----------------------------------------------------------------
    logger.info("=== Sample of collected predictions ===")
    for i, prediction in enumerate(all_predictions[:5]):  # only first 5
        logger.info(f"Prediction {i+1}: {prediction}")

    if write_predictions_file:
        os.makedirs(os.path.dirname(write_predictions_file), exist_ok=True)
        logger.info(f"Writing all predictions to {write_predictions_file}")
        try:
            with open(write_predictions_file, "w", encoding="utf-8") as out_f:
                for pred in all_predictions:
                    out_f.write(pred.strip() + "\n")
            logger.info("Finished writing predictions.")
        except Exception as e:
            logger.exception(f"Error writing predictions file: {str(e)}")

    logger.info("=== read_predictions function complete ===")

    return all_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read predictions from .pt files and optionally write them to a text file."
    )
    parser.add_argument(
        "--read_output_dir",
        type=str,
        default=DEFAULT_READ_OUTPUT_DIR,
        help="Directory to scan for .pt files."
    )
    parser.add_argument(
        "--max_predictions",
        type=int,
        default=DEFAULT_MAX_PREDICTIONS,
        help="Maximum number of predictions to collect."
    )
    parser.add_argument(
        "--write_predictions_file",
        type=str,
        default="",
        help="Optional file path to write predictions to. Leave blank for no file output."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help="Log file path."
    )
    args = parser.parse_args()

    # If an empty string is passed for write_predictions_file, set it to None
    write_file = args.write_predictions_file if args.write_predictions_file else None

    # Setup a basic logger if needed
    logging.basicConfig(level=logging.INFO)
    predictions = read_predictions(
        read_output_dir=args.read_output_dir,
        max_predictions=args.max_predictions,
        write_predictions_file=write_file,
        log_file=args.log_file
    )

    # Optionally, print the number of predictions collected when running from command line
    print(f"Collected {len(predictions)} predictions.")
