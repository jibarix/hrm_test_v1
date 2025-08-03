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
    inputs = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__inputs.json"))
    labels = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__labels.json"))
    puzzle_identifiers = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__puzzle_identifiers.json"))
    puzzle_indices = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__puzzle_indices.json"))
    group_indices = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__group_indices.json"))
    metadata = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_dataset.json"))
    
    # Convert to numpy arrays
    data = {
        "inputs": np.array(inputs, dtype=np.int32),
        "labels": np.array(labels, dtype=np.int32),  
        "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "group_indices": np.array(group_indices, dtype=np.int32)
    }
    
    # Create train/test split (80/20)
    split_idx = int(0.8 * len(data["inputs"]))
    
    for split_name, start_idx, end_idx in [("train", 0, split_idx), ("test", split_idx, len(data["inputs"]))]:
        save_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Slice data for this split
        split_data = {k: v[start_idx:end_idx] for k, v in data.items() if k not in ["puzzle_indices", "group_indices"]}
        
        # Adjust indices
        split_data["puzzle_indices"] = data["puzzle_indices"] - start_idx
        split_data["group_indices"] = data["group_indices"] - start_idx
        
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