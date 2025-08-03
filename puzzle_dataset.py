import os
import json

import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata


def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int, max_dataset_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Generate sample indices with bounds checking
        sample_indices = rng.choice(puzzle_size, append_size, replace=False)
        absolute_indices = puzzle_start + sample_indices
        
        # Add bounds checking to prevent index out of bounds errors
        valid_indices = absolute_indices[absolute_indices < max_dataset_size]
        if len(valid_indices) < append_size:
            # If we don't have enough valid indices, sample with replacement from valid range
            valid_range = min(puzzle_size, max_dataset_size - puzzle_start)
            if valid_range > 0:
                sample_indices = rng.choice(valid_range, append_size, replace=True)
                absolute_indices = puzzle_start + sample_indices
            else:
                # Skip this puzzle if no valid indices
                continue

        # Put into batch
        batch_puzzle_indices.append(np.full(len(absolute_indices), puzzle_id, dtype=np.int32))
        batch.append(absolute_indices)

        current_size += len(absolute_indices)

    if len(batch) == 0:
        return start_index, np.array([]), np.array([])
    
    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool

    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.

    rank: int
    num_replicas: int


class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()
        
        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self) -> PuzzleDatasetMetadata:
        with open(os.path.join(self.config.dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets:
            # Load subset
            self._data[set_name] = {
                field_name: np.load(os.path.join(self.config.dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                for field_name, mmap_mode in field_mmap_modes.items()
            }

    def _collate_batch(self, batch):
        # Convert dtype
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size

            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,

                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # To tensor
        return {k: torch.from_numpy(v) for k, v in batch.items()}
    
    def _iter_test(self):
        for set_name, dataset in self._data.items():  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                })

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Get dataset size for bounds checking
            max_dataset_size = len(dataset["inputs"])

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                    max_dataset_size=max_dataset_size,  # Add bounds checking
                )

                # Skip if no valid batch was generated
                if len(batch_indices) == 0:
                    continue

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                
                # Additional bounds check before accessing data
                valid_batch_indices = batch_indices[batch_indices < max_dataset_size]
                if len(valid_batch_indices) != len(batch_indices):
                    print(f"Warning: Some batch indices were out of bounds. Truncating batch from {len(batch_indices)} to {len(valid_batch_indices)}")
                    # Adjust both arrays to same length
                    min_len = min(len(valid_batch_indices), len(batch_puzzle_indices))
                    batch_indices = valid_batch_indices[:min_len]
                    batch_puzzle_indices = batch_puzzle_indices[:min_len]
                
                if len(batch_indices) == 0:
                    continue
                    
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                })

                yield set_name, batch, global_effective_batch_size
                
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()