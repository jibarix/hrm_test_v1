#!/usr/bin/env python3
"""
HRM Paper-Compatible Dataset Converter
Converts JSON files from paper-compatible logistics game to .npy format for HRM training.
"""

from typing import Optional
import os
import json
import numpy as np
from pathlib import Path

from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata

cli = ArgParser()

class PaperDataProcessConfig(BaseModel):
    source_dir: str = "dataset/raw-data/CitySimulator"  # Where JSON files are saved
    output_dir: str = "data/logistics-routing-1k"
    
    # Paper validation options
    validate_methodology: bool = True
    strict_paper_compliance: bool = True

def load_paper_compatible_data(source_dir: str):
    """Load all paper-compatible dataset files"""
    
    source_path = Path(source_dir)
    
    # Expected files from paper-compatible generator (18 total)
    expected_files = {
        # Train files (8)
        'train_dataset': 'nyc_routing_paper_train_dataset.json',
        'train_inputs': 'nyc_routing_paper_train_all__inputs.json',
        'train_labels': 'nyc_routing_paper_train_all__labels.json',
        'train_puzzle_identifiers': 'nyc_routing_paper_train_all__puzzle_identifiers.json',
        'train_puzzle_indices': 'nyc_routing_paper_train_all__puzzle_indices.json',
        'train_group_indices': 'nyc_routing_paper_train_all__group_indices.json',
        'train_base_scenario_ids': 'nyc_routing_paper_train_all__base_scenario_ids.json',
        'train_vehicle_types': 'nyc_routing_paper_train_all__vehicle_types.json',
        
        # Test files (8)
        'test_dataset': 'nyc_routing_paper_test_dataset.json',
        'test_inputs': 'nyc_routing_paper_test_all__inputs.json',
        'test_labels': 'nyc_routing_paper_test_all__labels.json',
        'test_puzzle_identifiers': 'nyc_routing_paper_test_all__puzzle_identifiers.json',
        'test_puzzle_indices': 'nyc_routing_paper_test_all__puzzle_indices.json',
        'test_group_indices': 'nyc_routing_paper_test_all__group_indices.json',
        'test_base_scenario_ids': 'nyc_routing_paper_test_all__base_scenario_ids.json',
        'test_vehicle_types': 'nyc_routing_paper_test_all__vehicle_types.json',
        
        # Global files (2)
        'identifiers': 'nyc_routing_paper_identifiers.json',
        'dataset_summary': 'nyc_routing_paper_dataset_summary.json'
    }
    
    # Check for missing files
    missing_files = []
    for file_key, filename in expected_files.items():
        if not (source_path / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"⚠️ Missing files: {missing_files}")
        print("Make sure you've downloaded all 18 files from the paper-compatible generator")
    
    # Load all available files
    data = {}
    for file_key, filename in expected_files.items():
        file_path = source_path / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                data[file_key] = json.load(f)
            print(f"✅ Loaded {filename}")
        else:
            print(f"❌ Missing {filename}")
    
    return data

def validate_paper_methodology(data: dict, strict: bool = True):
    """Validate that data follows HRM paper methodology"""
    
    print("\n🧠 Validating HRM Paper Methodology...")
    
    issues = []
    
    # Check expected example counts
    if 'train_inputs' in data and 'test_inputs' in data:
        train_count = len(data['train_inputs'])
        test_count = len(data['test_inputs'])
        
        expected_train = 960  # 240 base × 4 vehicles
        expected_test = 400   # 100 base × 4 vehicles
        
        if train_count != expected_train:
            issues.append(f"Train count mismatch: {train_count} vs expected {expected_train}")
        
        if test_count != expected_test:
            issues.append(f"Test count mismatch: {test_count} vs expected {expected_test}")
        
        print(f"  Example counts: {train_count} train, {test_count} test")
    
    # Validate base scenario separation
    if 'train_base_scenario_ids' in data and 'test_base_scenario_ids' in data:
        train_bases = set(data['train_base_scenario_ids'])
        test_bases = set(data['test_base_scenario_ids'])
        
        overlap = train_bases.intersection(test_bases)
        if overlap:
            issues.append(f"🚨 CRITICAL: Base scenario overlap: {len(overlap)} scenarios")
            if len(overlap) <= 10:
                issues.append(f"Overlapping IDs: {sorted(list(overlap))}")
        
        print(f"  Base scenarios: {len(train_bases)} train, {len(test_bases)} test, {len(overlap)} overlap")
    
    # Validate systematic vehicle augmentation
    if 'train_vehicle_types' in data and 'test_vehicle_types' in data:
        expected_vehicles = {'easy', 'normal', 'hard', 'expert'}
        
        train_vehicles = set(data['train_vehicle_types'])
        test_vehicles = set(data['test_vehicle_types'])
        
        missing_train = expected_vehicles - train_vehicles
        missing_test = expected_vehicles - test_vehicles
        
        if missing_train:
            issues.append(f"Missing train vehicles: {missing_train}")
        if missing_test:
            issues.append(f"Missing test vehicles: {missing_test}")
        
        # Check vehicle distribution
        from collections import Counter
        train_dist = Counter(data['train_vehicle_types'])
        test_dist = Counter(data['test_vehicle_types'])
        
        print(f"  Train vehicle distribution: {dict(train_dist)}")
        print(f"  Test vehicle distribution: {dict(test_dist)}")
        
        # Each vehicle should appear roughly equally (240/4=60 train, 100/4=25 test)
        expected_train_per_vehicle = 240
        expected_test_per_vehicle = 100
        
        for vehicle in expected_vehicles:
            train_vehicle_count = train_dist.get(vehicle, 0)
            test_vehicle_count = test_dist.get(vehicle, 0)
            
            # Allow some variance due to failed path generation
            if train_vehicle_count < expected_train_per_vehicle * 0.8:
                issues.append(f"Low train count for {vehicle}: {train_vehicle_count}")
            if test_vehicle_count < expected_test_per_vehicle * 0.8:
                issues.append(f"Low test count for {vehicle}: {test_vehicle_count}")
    
    # Check dataset summary for paper compliance
    if 'dataset_summary' in data:
        summary = data['dataset_summary']
        
        if summary.get('method') != 'paper_compatible_generation':
            issues.append("Dataset not generated with paper-compatible method")
        
        if summary.get('systematic_augmentation') != '4_vehicle_variants_per_base_scenario':
            issues.append("Systematic augmentation not properly configured")
        
        if summary.get('no_overlap') != 'train_test_base_scenarios_completely_separate':
            issues.append("Train/test separation not properly ensured")
    
    # Report validation results
    if issues:
        print(f"⚠️ Found {len(issues)} methodology issues:")
        for issue in issues:
            print(f"    {issue}")
        
        if strict:
            raise ValueError("Dataset fails strict paper methodology validation")
        else:
            print("Continuing with warnings...")
    else:
        print("✅ Dataset passes paper methodology validation")
    
    return len(issues) == 0

def convert_paper_dataset(config: PaperDataProcessConfig):
    """Convert paper-compatible dataset to HRM .npy format"""
    
    print(f"🧠 Converting HRM Paper-Compatible Dataset")
    print(f"Source: {config.source_dir}")
    print(f"Output: {config.output_dir}")
    
    # Load all data
    data = load_paper_compatible_data(config.source_dir)
    
    # Validate methodology if enabled
    if config.validate_methodology:
        validate_paper_methodology(data, strict=config.strict_paper_compliance)
    
    # Create output directories
    train_dir = Path(config.output_dir) / "train"
    test_dir = Path(config.output_dir) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert train split
    if all(key in data for key in ['train_inputs', 'train_labels', 'train_puzzle_identifiers', 
                                   'train_puzzle_indices', 'train_group_indices']):
        
        print("\n📊 Converting train split...")
        
        # Save standard HRM files
        np.save(train_dir / "all__inputs.npy", np.array(data['train_inputs'], dtype=np.int32))
        np.save(train_dir / "all__labels.npy", np.array(data['train_labels'], dtype=np.int32))
        np.save(train_dir / "all__puzzle_identifiers.npy", np.array(data['train_puzzle_identifiers'], dtype=np.int32))
        np.save(train_dir / "all__puzzle_indices.npy", np.array(data['train_puzzle_indices'], dtype=np.int32))
        np.save(train_dir / "all__group_indices.npy", np.array(data['train_group_indices'], dtype=np.int32))
        
        # Save paper-specific metadata as JSON (for analysis)
        if 'train_base_scenario_ids' in data:
            with open(train_dir / "all__base_scenario_ids.json", 'w') as f:
                json.dump(data['train_base_scenario_ids'], f)
        
        if 'train_vehicle_types' in data:
            with open(train_dir / "all__vehicle_types.json", 'w') as f:
                json.dump(data['train_vehicle_types'], f)
        
        # Create train metadata
        train_metadata = PuzzleDatasetMetadata(
            seq_len=1600,  # 40×40 grid
            vocab_size=10,  # 0-9 tokens
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=len(data['train_group_indices']) - 1,
            mean_puzzle_examples=len(data['train_inputs']) / (len(data['train_group_indices']) - 1),
            sets=["all"]
        )
        
        with open(train_dir / "dataset.json", 'w') as f:
            json.dump(train_metadata.model_dump(), f, indent=2)
        
        print(f"  ✅ Saved {len(data['train_inputs'])} train examples")
    
    # Convert test split
    if all(key in data for key in ['test_inputs', 'test_labels', 'test_puzzle_identifiers', 
                                   'test_puzzle_indices', 'test_group_indices']):
        
        print("\n📊 Converting test split...")
        
        # Save standard HRM files
        np.save(test_dir / "all__inputs.npy", np.array(data['test_inputs'], dtype=np.int32))
        np.save(test_dir / "all__labels.npy", np.array(data['test_labels'], dtype=np.int32))
        np.save(test_dir / "all__puzzle_identifiers.npy", np.array(data['test_puzzle_identifiers'], dtype=np.int32))
        np.save(test_dir / "all__puzzle_indices.npy", np.array(data['test_puzzle_indices'], dtype=np.int32))
        np.save(test_dir / "all__group_indices.npy", np.array(data['test_group_indices'], dtype=np.int32))
        
        # Save paper-specific metadata as JSON (for analysis)
        if 'test_base_scenario_ids' in data:
            with open(test_dir / "all__base_scenario_ids.json", 'w') as f:
                json.dump(data['test_base_scenario_ids'], f)
        
        if 'test_vehicle_types' in data:
            with open(test_dir / "all__vehicle_types.json", 'w') as f:
                json.dump(data['test_vehicle_types'], f)
        
        # Create test metadata
        test_metadata = PuzzleDatasetMetadata(
            seq_len=1600,  # 40×40 grid
            vocab_size=10,  # 0-9 tokens
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=len(data['test_group_indices']) - 1,
            mean_puzzle_examples=len(data['test_inputs']) / (len(data['test_group_indices']) - 1),
            sets=["all"]
        )
        
        with open(test_dir / "dataset.json", 'w') as f:
            json.dump(test_metadata.model_dump(), f, indent=2)
        
        print(f"  ✅ Saved {len(data['test_inputs'])} test examples")
    
    # Save global identifiers
    if 'identifiers' in data:
        with open(Path(config.output_dir) / "identifiers.json", 'w') as f:
            json.dump(data['identifiers'], f)
    
    # Save dataset summary for reference
    if 'dataset_summary' in data:
        with open(Path(config.output_dir) / "dataset_summary.json", 'w') as f:
            json.dump(data['dataset_summary'], f, indent=2)
    
    print(f"\n✅ Paper-compatible dataset conversion complete!")
    print(f"📁 Output saved to: {config.output_dir}")
    print("\n🧠 Next steps:")
    print("  1. Run paper analysis: python paper_data_analysis.py")  
    print("  2. Train HRM: python pretrain.py data_path=data/logistics-routing-1k")

@cli.command(singleton=True)
def main(config: PaperDataProcessConfig):
    convert_paper_dataset(config)

if __name__ == "__main__":
    cli()