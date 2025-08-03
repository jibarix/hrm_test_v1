from typing import Optional
import os
import json
import numpy as np
from pathlib import Path

from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    source_dir: str = "dataset/raw-data/Logistics"  # Where you saved the JSON files
    output_dir: str = "data/logistics-routing-1k"
    
def convert_dataset(config: DataProcessConfig):
    # Read the JSON files we generated
    inputs_data = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__inputs.json"))
    labels_data = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__labels.json"))
    puzzle_identifiers_data = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__puzzle_identifiers.json"))
    puzzle_indices_data = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__puzzle_indices.json"))
    group_indices_data = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__group_indices.json"))
    metadata = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_dataset.json"))
    
    # Convert to numpy arrays
    data = {
        "inputs": np.array(inputs_data, dtype=np.int32),
        "labels": np.array(labels_data, dtype=np.int32),  
        "puzzle_identifiers": np.array(puzzle_identifiers_data, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices_data, dtype=np.int32),
        "group_indices": np.array(group_indices_data, dtype=np.int32)
    }
    
    # Create train/test split (80/20)
    num_examples = len(data["inputs"])
    split_idx = int(0.8 * num_examples)
    
    # Determine the puzzle index where the split occurs
    puzzle_split_idx = np.searchsorted(data["puzzle_indices"], split_idx, side='right')
    
    # Ensure groups are not split across train/test
    group_split_idx = np.searchsorted(data["group_indices"], puzzle_split_idx, side='right')

    for split_name, start_puzzle, end_puzzle in [("train", 0, puzzle_split_idx), ("test", puzzle_split_idx, len(data["puzzle_identifiers"]))]:
        save_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(save_dir, exist_ok=True)
        
        start_example = data["puzzle_indices"][start_puzzle]
        end_example = data["puzzle_indices"][end_puzzle]

        # Slice data for this split
        split_data = {
            "inputs": data["inputs"][start_example:end_example],
            "labels": data["labels"][start_example:end_example],
            "puzzle_identifiers": data["puzzle_identifiers"][start_puzzle:end_puzzle]
        }
        
        # Adjust indices
        split_data["puzzle_indices"] = data["puzzle_indices"][start_puzzle:end_puzzle+1] - start_example
        
        start_group = np.searchsorted(data["group_indices"], start_puzzle, side='right')-1
        end_group = np.searchsorted(data["group_indices"], end_puzzle, side='right')-1
        split_data["group_indices"] = data["group_indices"][start_group:end_group+1] - start_puzzle
        
        # Save .npy files
        for k, v in split_data.items():
            np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
            
        # Create metadata
        split_metadata = PuzzleDatasetMetadata(
            seq_len=metadata["seq_len"],
            vocab_size=metadata["vocab_size"],
            pad_id=metadata["pad_id"],
            ignore_label_id=metadata["ignore_label_id"],
            blank_identifier_id=metadata["blank_identifier_id"],
            num_puzzle_identifiers=metadata["num_puzzle_identifiers"],
            total_groups=len(split_data["group_indices"]) - 1,
            mean_puzzle_examples=1.0,
            sets=["all"]
        )
        
        # Save metadata
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(split_metadata.model_dump(), f)
    
    # Save identifiers
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)

if __name__ == "__main__":
    cli()