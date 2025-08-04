#!/usr/bin/env python3
"""
Comprehensive Data Quality Analysis for Simple Pathfinding Dataset
Analyzes the actual generated data to verify the issues found in code review.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
from pathlib import Path

# Token mapping from the HTML
HRM_TOKEN_MAP = {
    'PAD': 0,
    'OBSTACLE': 1,
    'ROAD': 2,
    'SLOW_ROAD': 3,
    'START': 4,
    'END': 5,
    'PATH': 6
}

MAP_DIMENSIONS = {'width': 40, 'height': 40}

class DataAnalyzer:
    def __init__(self, data_dir="data/simple-pathfinding-1k"):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        
        # Load data
        self.train_inputs = np.load(self.train_dir / "all__inputs.npy")
        self.train_labels = np.load(self.train_dir / "all__labels.npy")
        self.test_inputs = np.load(self.test_dir / "all__inputs.npy") 
        self.test_labels = np.load(self.test_dir / "all__labels.npy")
        
        print(f"Loaded {len(self.train_inputs)} training examples")
        print(f"Loaded {len(self.test_inputs)} test examples")
        
        self.results = {}
    
    def tokens_to_grid(self, tokens):
        """Convert 1600-token sequence back to 40x40 grid"""
        return tokens.reshape(MAP_DIMENSIONS['height'], MAP_DIMENSIONS['width'])
    
    def analyze_uniqueness(self):
        """Check for duplicate examples and uniqueness issues"""
        print("\n" + "="*60)
        print("1. UNIQUENESS ANALYSIS")
        print("="*60)
        
        # Check for exact input duplicates
        unique_inputs, indices, counts = np.unique(self.train_inputs, axis=0, return_inverse=True, return_counts=True)
        input_duplicates = np.sum(counts > 1)
        
        # Check for exact label duplicates  
        unique_labels, _, label_counts = np.unique(self.train_labels, axis=0, return_inverse=True, return_counts=True)
        label_duplicates = np.sum(label_counts > 1)
        
        # Check for exact input-output pair duplicates
        combined = np.concatenate([self.train_inputs, self.train_labels], axis=1)
        unique_pairs, _, pair_counts = np.unique(combined, axis=0, return_inverse=True, return_counts=True)
        pair_duplicates = np.sum(pair_counts > 1)
        
        # Train/test leakage check
        train_set = set(tuple(x) for x in self.train_inputs)
        test_set = set(tuple(x) for x in self.test_inputs)
        leakage = len(train_set.intersection(test_set))
        
        print(f"Training examples: {len(self.train_inputs)}")
        print(f"Unique input maps: {len(unique_inputs)} ({len(unique_inputs)/len(self.train_inputs)*100:.1f}%)")
        print(f"Input duplicates: {input_duplicates} examples appear multiple times")
        print(f"Unique output paths: {len(unique_labels)} ({len(unique_labels)/len(self.train_labels)*100:.1f}%)")
        print(f"Label duplicates: {label_duplicates} paths appear multiple times")
        print(f"Unique input-output pairs: {len(unique_pairs)} ({len(unique_pairs)/len(self.train_inputs)*100:.1f}%)")
        print(f"🚨 EXACT DUPLICATES: {pair_duplicates} complete examples duplicated")
        print(f"🚨 TRAIN/TEST LEAKAGE: {leakage} examples appear in both sets")
        
        # Most frequent duplicate
        if pair_duplicates > 0:
            max_count_idx = np.argmax(pair_counts)
            print(f"Most frequent example appears {pair_counts[max_count_idx]} times")
        
        self.results['uniqueness'] = {
            'unique_inputs': len(unique_inputs),
            'unique_labels': len(unique_labels),
            'unique_pairs': len(unique_pairs),
            'input_duplicates': input_duplicates,
            'label_duplicates': label_duplicates,
            'pair_duplicates': pair_duplicates,
            'train_test_leakage': leakage
        }
    
    def analyze_start_end_positions(self):
        """Analyze start/end position distributions"""
        print("\n" + "="*60)
        print("2. START/END POSITION ANALYSIS")
        print("="*60)
        
        start_positions = []
        end_positions = []
        distances = []
        edge_distances_start = []
        edge_distances_end = []
        
        for i, input_tokens in enumerate(self.train_inputs):
            grid = self.tokens_to_grid(input_tokens)
            
            # Find start and end positions
            start_pos = np.where(grid == HRM_TOKEN_MAP['START'])
            end_pos = np.where(grid == HRM_TOKEN_MAP['END'])
            
            if len(start_pos[0]) > 0 and len(end_pos[0]) > 0:
                start_y, start_x = start_pos[0][0], start_pos[1][0]
                end_y, end_x = end_pos[0][0], end_pos[1][0]
                
                start_positions.append((start_x, start_y))
                end_positions.append((end_x, end_y))
                
                # Manhattan distance
                manhattan_dist = abs(start_x - end_x) + abs(start_y - end_y)
                distances.append(manhattan_dist)
                
                # Distance from edges
                start_edge_dist = min(start_x, start_y, 
                                    MAP_DIMENSIONS['width'] - 1 - start_x, 
                                    MAP_DIMENSIONS['height'] - 1 - start_y)
                end_edge_dist = min(end_x, end_y,
                                  MAP_DIMENSIONS['width'] - 1 - end_x,
                                  MAP_DIMENSIONS['height'] - 1 - end_y)
                
                edge_distances_start.append(start_edge_dist)
                edge_distances_end.append(end_edge_dist)
        
        # Analysis
        distances = np.array(distances)
        edge_distances_start = np.array(edge_distances_start)
        edge_distances_end = np.array(edge_distances_end)
        
        print(f"Manhattan distances - Min: {distances.min()}, Max: {distances.max()}, Mean: {distances.mean():.1f}")
        print(f"Distance std dev: {distances.std():.1f} (low std = predictable)")
        print(f"🚨 Distances 15-35 range: {np.sum((distances >= 15) & (distances <= 35))/len(distances)*100:.1f}% of examples")
        print(f"Start edge distances - Min: {edge_distances_start.min()}, Max: {edge_distances_start.max()}")
        print(f"🚨 Start positions ≥2 from edge: {np.sum(edge_distances_start >= 2)/len(edge_distances_start)*100:.1f}%")
        print(f"End edge distances - Min: {edge_distances_end.min()}, Max: {edge_distances_end.max()}")
        print(f"🚨 End positions ≥2 from edge: {np.sum(edge_distances_end >= 2)/len(edge_distances_end)*100:.1f}%")
        
        # Check for position clustering
        start_x_coords = [pos[0] for pos in start_positions]
        start_y_coords = [pos[1] for pos in start_positions]
        
        print(f"Start X spread: {np.std(start_x_coords):.1f} (max possible: ~11.5)")
        print(f"Start Y spread: {np.std(start_y_coords):.1f} (max possible: ~11.5)")
        
        self.results['positions'] = {
            'distance_range': (distances.min(), distances.max()),
            'distance_mean': distances.mean(),
            'distance_std': distances.std(),
            'constrained_distances': np.sum((distances >= 15) & (distances <= 35))/len(distances),
            'start_edge_constrained': np.sum(edge_distances_start >= 2)/len(edge_distances_start),
            'end_edge_constrained': np.sum(edge_distances_end >= 2)/len(edge_distances_end)
        }
        
        return distances, start_positions, end_positions
    
    def analyze_map_structure(self):
        """Analyze map structural diversity"""
        print("\n" + "="*60)
        print("3. MAP STRUCTURE ANALYSIS") 
        print("="*60)
        
        obstacle_ratios = []
        slow_road_ratios = []
        connectivity_scores = []
        
        for input_tokens in self.train_inputs:
            grid = self.tokens_to_grid(input_tokens)
            
            # Count tile types (excluding start/end)
            total_tiles = MAP_DIMENSIONS['width'] * MAP_DIMENSIONS['height']
            obstacles = np.sum(grid == HRM_TOKEN_MAP['OBSTACLE'])
            roads = np.sum(grid == HRM_TOKEN_MAP['ROAD'])
            slow_roads = np.sum(grid == HRM_TOKEN_MAP['SLOW_ROAD'])
            
            obstacle_ratio = obstacles / total_tiles
            slow_road_ratio = slow_roads / (roads + slow_roads) if (roads + slow_roads) > 0 else 0
            
            obstacle_ratios.append(obstacle_ratio)
            slow_road_ratios.append(slow_road_ratio)
        
        obstacle_ratios = np.array(obstacle_ratios)
        slow_road_ratios = np.array(slow_road_ratios)
        
        print(f"Obstacle ratios - Min: {obstacle_ratios.min():.3f}, Max: {obstacle_ratios.max():.3f}, Mean: {obstacle_ratios.mean():.3f}")
        print(f"Obstacle ratio std: {obstacle_ratios.std():.3f} (target: diverse)")
        print(f"🚨 Examples with ~30% obstacles: {np.sum(np.abs(obstacle_ratios - 0.3) < 0.05)/len(obstacle_ratios)*100:.1f}%")
        
        print(f"Slow road ratios - Min: {slow_road_ratios.min():.3f}, Max: {slow_road_ratios.max():.3f}, Mean: {slow_road_ratios.mean():.3f}")
        print(f"Slow road ratio std: {slow_road_ratios.std():.3f}")
        print(f"🚨 Examples with ~20% slow roads: {np.sum(np.abs(slow_road_ratios - 0.2) < 0.05)/len(slow_road_ratios)*100:.1f}%")
        
        self.results['structure'] = {
            'obstacle_ratio_mean': obstacle_ratios.mean(),
            'obstacle_ratio_std': obstacle_ratios.std(),
            'slow_road_ratio_mean': slow_road_ratios.mean(),
            'slow_road_ratio_std': slow_road_ratios.std(),
            'fixed_obstacle_ratio': np.sum(np.abs(obstacle_ratios - 0.3) < 0.05)/len(obstacle_ratios),
            'fixed_slow_ratio': np.sum(np.abs(slow_road_ratios - 0.2) < 0.05)/len(slow_road_ratios)
        }
        
        return obstacle_ratios, slow_road_ratios
    
    def analyze_path_characteristics(self):
        """Analyze path diversity and complexity"""
        print("\n" + "="*60)
        print("4. PATH CHARACTERISTICS ANALYSIS")
        print("="*60)
        
        path_lengths = []
        path_complexities = []  # Number of direction changes
        
        for label_tokens in self.train_labels:
            grid = self.tokens_to_grid(label_tokens)
            path_positions = np.where(grid == HRM_TOKEN_MAP['PATH'])
            
            if len(path_positions[0]) > 0:
                path_length = len(path_positions[0])
                path_lengths.append(path_length)
                
                # Calculate path complexity (direction changes)
                if path_length > 2:
                    positions = list(zip(path_positions[1], path_positions[0]))  # (x, y) pairs
                    positions = sorted(positions)  # Sort to get path order (approximate)
                    
                    direction_changes = 0
                    prev_direction = None
                    
                    for i in range(1, len(positions)):
                        dx = positions[i][0] - positions[i-1][0]
                        dy = positions[i][1] - positions[i-1][1]
                        current_direction = (dx, dy)
                        
                        if prev_direction and current_direction != prev_direction:
                            direction_changes += 1
                        prev_direction = current_direction
                    
                    path_complexities.append(direction_changes)
        
        path_lengths = np.array(path_lengths)
        path_complexities = np.array(path_complexities)
        
        print(f"Path lengths - Min: {path_lengths.min()}, Max: {path_lengths.max()}, Mean: {path_lengths.mean():.1f}")
        print(f"Path length std: {path_lengths.std():.1f}")
        print(f"🚨 Paths length 15-35: {np.sum((path_lengths >= 15) & (path_lengths <= 35))/len(path_lengths)*100:.1f}%")
        
        if len(path_complexities) > 0:
            print(f"Path complexities (direction changes) - Mean: {path_complexities.mean():.1f}, Std: {path_complexities.std():.1f}")
        
        # Check for identical paths
        unique_paths = np.unique(self.train_labels, axis=0)
        path_duplicates = len(self.train_labels) - len(unique_paths)
        print(f"🚨 Identical paths: {path_duplicates} ({path_duplicates/len(self.train_labels)*100:.1f}%)")
        
        self.results['paths'] = {
            'length_range': (path_lengths.min(), path_lengths.max()),
            'length_mean': path_lengths.mean(),
            'length_std': path_lengths.std(),
            'constrained_lengths': np.sum((path_lengths >= 15) & (path_lengths <= 35))/len(path_lengths),
            'identical_paths': path_duplicates/len(self.train_labels)
        }
        
        return path_lengths
    
    def analyze_a_star_quality(self):
        """Check for degenerate A* solutions"""
        print("\n" + "="*60)
        print("5. A* SOLUTION QUALITY ANALYSIS")
        print("="*60)
        
        valid_paths = 0
        invalid_paths = 0
        path_costs = []
        
        for i, (input_tokens, label_tokens) in enumerate(zip(self.train_inputs[:50], self.train_labels[:50])):  # Sample first 50
            input_grid = self.tokens_to_grid(input_tokens)
            label_grid = self.tokens_to_grid(label_tokens)
            
            # Find start, end, and path
            start_pos = np.where(input_grid == HRM_TOKEN_MAP['START'])
            end_pos = np.where(input_grid == HRM_TOKEN_MAP['END'])
            path_pos = np.where(label_grid == HRM_TOKEN_MAP['PATH'])
            
            if len(start_pos[0]) > 0 and len(end_pos[0]) > 0 and len(path_pos[0]) > 0:
                start = (start_pos[1][0], start_pos[0][0])  # (x, y)
                end = (end_pos[1][0], end_pos[0][0])
                path_positions = list(zip(path_pos[1], path_pos[0]))
                
                # Check if path is valid (connected and reaches start/end)
                path_set = set(path_positions)
                if start in path_set and end in path_set:
                    # Calculate path cost
                    total_cost = 0
                    for x, y in path_positions:
                        if input_grid[y, x] == HRM_TOKEN_MAP['SLOW_ROAD']:
                            total_cost += 2
                        else:
                            total_cost += 1
                    
                    path_costs.append(total_cost)
                    valid_paths += 1
                else:
                    invalid_paths += 1
        
        print(f"Valid paths: {valid_paths}")
        print(f"🚨 Invalid paths: {invalid_paths}")
        
        if path_costs:
            path_costs = np.array(path_costs)
            print(f"Path costs - Min: {path_costs.min()}, Max: {path_costs.max()}, Mean: {path_costs.mean():.1f}")
        
        self.results['a_star'] = {
            'valid_paths': valid_paths,
            'invalid_paths': invalid_paths,
            'path_costs_mean': np.mean(path_costs) if len(path_costs) > 0 else 0
        }
    
    def generate_visualizations(self, distances, obstacle_ratios, path_lengths):
        """Generate visualization plots"""
        print("\n" + "="*60)
        print("6. GENERATING VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Data Quality Analysis Results', fontsize=16)
        
        # Distance distribution
        axes[0,0].hist(distances, bins=20, alpha=0.7, color='blue')
        axes[0,0].axvline(15, color='red', linestyle='--', label='Min constraint (15)')
        axes[0,0].axvline(35, color='red', linestyle='--', label='Max constraint (35)')
        axes[0,0].set_title('Start-End Manhattan Distances')
        axes[0,0].set_xlabel('Distance')
        axes[0,0].set_ylabel('Count')
        axes[0,0].legend()
        
        # Obstacle ratio distribution
        axes[0,1].hist(obstacle_ratios, bins=20, alpha=0.7, color='brown')
        axes[0,1].axvline(0.3, color='red', linestyle='--', label='Target 30%')
        axes[0,1].set_title('Obstacle Density Distribution')
        axes[0,1].set_xlabel('Obstacle Ratio')
        axes[0,1].set_ylabel('Count')
        axes[0,1].legend()
        
        # Path length distribution
        axes[0,2].hist(path_lengths, bins=20, alpha=0.7, color='green')
        axes[0,2].set_title('Path Length Distribution')
        axes[0,2].set_xlabel('Path Length')
        axes[0,2].set_ylabel('Count')
        
        # Uniqueness metrics
        uniqueness_metrics = ['Input\nDuplicates', 'Label\nDuplicates', 'Pair\nDuplicates', 'Train/Test\nLeakage']
        uniqueness_values = [
            self.results['uniqueness']['input_duplicates'],
            self.results['uniqueness']['label_duplicates'], 
            self.results['uniqueness']['pair_duplicates'],
            self.results['uniqueness']['train_test_leakage']
        ]
        
        bars = axes[1,0].bar(uniqueness_metrics, uniqueness_values, color=['red' if v > 0 else 'green' for v in uniqueness_values])
        axes[1,0].set_title('Uniqueness Issues (0 = Good)')
        axes[1,0].set_ylabel('Count')
        
        # Constraint compliance
        constraint_metrics = ['Distance\n15-35', 'Start Edge\n≥2', 'End Edge\n≥2', 'Obstacle\n~30%']
        constraint_values = [
            self.results['positions']['constrained_distances'] * 100,
            self.results['positions']['start_edge_constrained'] * 100,
            self.results['positions']['end_edge_constrained'] * 100,
            self.results['structure']['fixed_obstacle_ratio'] * 100
        ]
        
        bars = axes[1,1].bar(constraint_metrics, constraint_values, color='orange')
        axes[1,1].set_title('Constraint Compliance (%)')
        axes[1,1].set_ylabel('Percentage')
        axes[1,1].axhline(90, color='red', linestyle='--', label='90% threshold')
        axes[1,1].legend()
        
        # Sample map visualization
        sample_input = self.tokens_to_grid(self.train_inputs[0])
        sample_label = self.tokens_to_grid(self.train_labels[0])
        
        # Create combined visualization
        combined_map = sample_input.copy().astype(float)
        path_mask = sample_label == HRM_TOKEN_MAP['PATH']
        combined_map[path_mask] = HRM_TOKEN_MAP['PATH']
        
        im = axes[1,2].imshow(combined_map, cmap='tab10', vmin=0, vmax=6)
        axes[1,2].set_title('Sample Map (First Example)')
        axes[1,2].set_xticks([])
        axes[1,2].set_yticks([])
        
        plt.tight_layout()
        plt.savefig('data_quality_analysis.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to: data_quality_analysis.png")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("7. COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        # Severity scoring
        severity_score = 0
        issues = []
        
        # Check uniqueness issues
        if self.results['uniqueness']['pair_duplicates'] > 0:
            severity_score += 10
            issues.append(f"🚨 CRITICAL: {self.results['uniqueness']['pair_duplicates']} exact duplicate examples")
        
        if self.results['uniqueness']['train_test_leakage'] > 0:
            severity_score += 10
            issues.append(f"🚨 CRITICAL: {self.results['uniqueness']['train_test_leakage']} examples in both train/test")
        
        # Check position constraints
        if self.results['positions']['constrained_distances'] > 0.8:
            severity_score += 8
            issues.append(f"🚨 HIGH: {self.results['positions']['constrained_distances']*100:.1f}% of examples have constrained distances (15-35)")
        
        if self.results['positions']['start_edge_constrained'] > 0.8:
            severity_score += 6
            issues.append(f"🚨 MEDIUM: {self.results['positions']['start_edge_constrained']*100:.1f}% of start positions avoid edges")
        
        # Check structural diversity
        if self.results['structure']['obstacle_ratio_std'] < 0.05:
            severity_score += 6
            issues.append(f"🚨 MEDIUM: Low obstacle diversity (std: {self.results['structure']['obstacle_ratio_std']:.3f})")
        
        if self.results['structure']['fixed_obstacle_ratio'] > 0.7:
            severity_score += 5
            issues.append(f"🚨 MEDIUM: {self.results['structure']['fixed_obstacle_ratio']*100:.1f}% examples have ~30% obstacles")
        
        # Check path diversity
        if self.results['paths']['identical_paths'] > 0.1:
            severity_score += 7
            issues.append(f"🚨 HIGH: {self.results['paths']['identical_paths']*100:.1f}% identical paths")
        
        print(f"SEVERITY SCORE: {severity_score}/50 (higher = worse)")
        print(f"RECOMMENDATION: {'🚨 REGENERATE DATASET' if severity_score > 25 else '⚠️ IMPROVE DIVERSITY' if severity_score > 15 else '✅ ACCEPTABLE'}")
        
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        
        if severity_score > 25:
            print("\n🔧 CRITICAL FIXES NEEDED:")
            print("  1. Add uniqueness checking to prevent duplicates")
            print("  2. Remove edge distance constraints for start/end positions") 
            print("  3. Vary obstacle density (20-40% instead of fixed 30%)")
            print("  4. Vary corridor count (0-5 instead of fixed 3)")
            print("  5. Expand distance constraints (10-50 instead of 15-35)")
        
        return severity_score, issues
    
    def run_full_analysis(self):
        """Run complete data quality analysis"""
        print("Starting comprehensive data quality analysis...")
        print(f"Dataset location: {self.data_dir}")
        
        try:
            # Run all analyses
            self.analyze_uniqueness()
            distances, start_pos, end_pos = self.analyze_start_end_positions()
            obstacle_ratios, slow_ratios = self.analyze_map_structure()
            path_lengths = self.analyze_path_characteristics()
            self.analyze_a_star_quality()
            
            # Generate visualizations
            self.generate_visualizations(distances, obstacle_ratios, path_lengths)
            
            # Generate final report
            severity_score, issues = self.generate_report()
            
            return severity_score < 25  # Return True if dataset is acceptable
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    analyzer = DataAnalyzer()
    is_acceptable = analyzer.run_full_analysis()
    
    if not is_acceptable:
        print("\n" + "="*60)
        print("CONCLUSION: Dataset quality issues likely causing overfitting!")
        print("Recommend regenerating data with fixes from code analysis.")
        print("="*60)