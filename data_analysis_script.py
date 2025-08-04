#!/usr/bin/env python3
"""
Enhanced Data Quality Analysis for City Logistics Routing Dataset
Analyzes the logistics game generated data to verify quality and diversity.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
from pathlib import Path

# Token mapping from the logistics game HTML
LOGISTICS_TOKEN_MAP = {
    'PAD': 0,
    'OBSTACLE': 1,      # Buildings/Central Park
    'SMALL_ROAD': 2,    # Side Streets  
    'LARGE_ROAD': 3,    # Major Avenues
    'BROADWAY': 4,      # Broadway diagonal
    'TRAFFIC_JAM': 5,   # Heavy Traffic
    'ROAD_CLOSURE': 6,  # Road Closure
    'START': 7,         # Start Point
    'END': 8,           # End Point
    'PATH': 9           # Optimal Route
}

MAP_DIMENSIONS = {'width': 40, 'height': 40}

class LogisticsDataAnalyzer:
    def __init__(self, data_dir="data/logistics-routing-1k"):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        
        # Load data
        try:
            self.train_inputs = np.load(self.train_dir / "all__inputs.npy")
            self.train_labels = np.load(self.train_dir / "all__labels.npy")
            self.test_inputs = np.load(self.test_dir / "all__inputs.npy") 
            self.test_labels = np.load(self.test_dir / "all__labels.npy")
            
            print(f"Loaded {len(self.train_inputs)} training examples")
            print(f"Loaded {len(self.test_inputs)} test examples")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Make sure the dataset has been converted from JSON to .npy format")
            return
        
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
        """Analyze start/end position distributions and distances"""
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
            start_pos = np.where(grid == LOGISTICS_TOKEN_MAP['START'])
            end_pos = np.where(grid == LOGISTICS_TOKEN_MAP['END'])
            
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
        
        distances = np.array(distances)
        edge_distances_start = np.array(edge_distances_start)
        edge_distances_end = np.array(edge_distances_end)
        
        print(f"Manhattan distances - Min: {distances.min()}, Max: {distances.max()}, Mean: {distances.mean():.1f}")
        print(f"Distance std dev: {distances.std():.1f}")
        print(f"🚨 Constrained distances (15-35): {np.sum((distances >= 15) & (distances <= 35))/len(distances)*100:.1f}% of examples")
        print(f"Start edge distances - Min: {edge_distances_start.min()}, Max: {edge_distances_start.max()}")
        print(f"End edge distances - Min: {edge_distances_end.min()}, Max: {edge_distances_end.max()}")
        
        # Check for position clustering
        start_x_coords = [pos[0] for pos in start_positions]
        start_y_coords = [pos[1] for pos in start_positions]
        
        print(f"Start X spread: {np.std(start_x_coords):.1f}")
        print(f"Start Y spread: {np.std(start_y_coords):.1f}")
        
        self.results['positions'] = {
            'distance_range': (distances.min(), distances.max()),
            'distance_mean': distances.mean(),
            'distance_std': distances.std(),
            'start_positions': start_positions,
            'end_positions': end_positions
        }
        
        return distances, start_positions, end_positions
    
    def analyze_road_structure(self):
        """Analyze city road structure diversity"""
        print("\n" + "="*60)
        print("3. CITY ROAD STRUCTURE ANALYSIS") 
        print("="*60)
        
        obstacle_ratios = []
        small_road_ratios = []
        large_road_ratios = []
        broadway_ratios = []
        traffic_ratios = []
        closure_ratios = []
        
        road_type_distributions = []
        
        for input_tokens in self.train_inputs:
            grid = self.tokens_to_grid(input_tokens)
            
            # Count each tile type
            total_tiles = MAP_DIMENSIONS['width'] * MAP_DIMENSIONS['height']
            
            obstacles = np.sum(grid == LOGISTICS_TOKEN_MAP['OBSTACLE'])
            small_roads = np.sum(grid == LOGISTICS_TOKEN_MAP['SMALL_ROAD'])
            large_roads = np.sum(grid == LOGISTICS_TOKEN_MAP['LARGE_ROAD'])
            broadway = np.sum(grid == LOGISTICS_TOKEN_MAP['BROADWAY'])
            traffic_jams = np.sum(grid == LOGISTICS_TOKEN_MAP['TRAFFIC_JAM'])
            road_closures = np.sum(grid == LOGISTICS_TOKEN_MAP['ROAD_CLOSURE'])
            
            total_roads = small_roads + large_roads + broadway + traffic_jams + road_closures
            
            obstacle_ratios.append(obstacles / total_tiles)
            
            if total_roads > 0:
                small_road_ratios.append(small_roads / total_roads)
                large_road_ratios.append(large_roads / total_roads)
                broadway_ratios.append(broadway / total_roads)
                traffic_ratios.append(traffic_jams / total_roads)
                closure_ratios.append(road_closures / total_roads)
            
            road_type_distributions.append({
                'obstacles': obstacles,
                'small_roads': small_roads,
                'large_roads': large_roads,
                'broadway': broadway,
                'traffic_jams': traffic_jams,
                'road_closures': road_closures
            })
        
        # Convert to numpy arrays
        obstacle_ratios = np.array(obstacle_ratios)
        small_road_ratios = np.array(small_road_ratios)
        large_road_ratios = np.array(large_road_ratios)
        broadway_ratios = np.array(broadway_ratios)
        traffic_ratios = np.array(traffic_ratios)
        closure_ratios = np.array(closure_ratios)
        
        print(f"Obstacle ratios - Min: {obstacle_ratios.min():.3f}, Max: {obstacle_ratios.max():.3f}, Mean: {obstacle_ratios.mean():.3f}")
        print(f"Small road ratios - Min: {small_road_ratios.min():.3f}, Max: {small_road_ratios.max():.3f}, Mean: {small_road_ratios.mean():.3f}")
        print(f"Large road ratios - Min: {large_road_ratios.min():.3f}, Max: {large_road_ratios.max():.3f}, Mean: {large_road_ratios.mean():.3f}")
        print(f"Broadway ratios - Min: {broadway_ratios.min():.3f}, Max: {broadway_ratios.max():.3f}, Mean: {broadway_ratios.mean():.3f}")
        print(f"Traffic jam ratios - Min: {traffic_ratios.min():.3f}, Max: {traffic_ratios.max():.3f}, Mean: {traffic_ratios.mean():.3f}")
        print(f"Road closure ratios - Min: {closure_ratios.min():.3f}, Max: {closure_ratios.max():.3f}, Mean: {closure_ratios.mean():.3f}")
        
        # Check for diversity
        print(f"Obstacle ratio std: {obstacle_ratios.std():.3f}")
        print(f"Road type diversity std: Small={small_road_ratios.std():.3f}, Large={large_road_ratios.std():.3f}, Broadway={broadway_ratios.std():.3f}")
        
        self.results['structure'] = {
            'obstacle_ratios': obstacle_ratios,
            'road_distributions': road_type_distributions,
            'diversity_metrics': {
                'obstacle_std': obstacle_ratios.std(),
                'small_road_std': small_road_ratios.std(),
                'large_road_std': large_road_ratios.std(),
                'broadway_std': broadway_ratios.std()
            }
        }
        
        return obstacle_ratios, road_type_distributions
    
    def analyze_vehicle_time_diversity(self):
        """Analyze vehicle type and time diversity (requires metadata)"""
        print("\n" + "="*60)
        print("4. VEHICLE & TIME DIVERSITY ANALYSIS")
        print("="*60)
        
        # This would require vehicle/time metadata from the generation process
        # For now, we can analyze traffic patterns as a proxy
        
        traffic_intensity_levels = []
        closure_patterns = []
        
        for input_tokens in self.train_inputs:
            grid = self.tokens_to_grid(input_tokens)
            
            total_roads = np.sum((grid == LOGISTICS_TOKEN_MAP['SMALL_ROAD']) | 
                               (grid == LOGISTICS_TOKEN_MAP['LARGE_ROAD']) | 
                               (grid == LOGISTICS_TOKEN_MAP['BROADWAY']))
            traffic_jams = np.sum(grid == LOGISTICS_TOKEN_MAP['TRAFFIC_JAM'])
            road_closures = np.sum(grid == LOGISTICS_TOKEN_MAP['ROAD_CLOSURE'])
            
            if total_roads > 0:
                traffic_intensity = traffic_jams / total_roads
                closure_intensity = road_closures / total_roads
                
                traffic_intensity_levels.append(traffic_intensity)
                closure_patterns.append(closure_intensity)
        
        traffic_intensity_levels = np.array(traffic_intensity_levels)
        closure_patterns = np.array(closure_patterns)
        
        print(f"Traffic intensity - Min: {traffic_intensity_levels.min():.3f}, Max: {traffic_intensity_levels.max():.3f}, Mean: {traffic_intensity_levels.mean():.3f}")
        print(f"Closure patterns - Min: {closure_patterns.min():.3f}, Max: {closure_patterns.max():.3f}, Mean: {closure_patterns.mean():.3f}")
        print(f"Traffic diversity (std): {traffic_intensity_levels.std():.3f}")
        print(f"Closure diversity (std): {closure_patterns.std():.3f}")
        
        # Check for time pattern diversity
        low_traffic = np.sum(traffic_intensity_levels < 0.1)
        med_traffic = np.sum((traffic_intensity_levels >= 0.1) & (traffic_intensity_levels < 0.3))
        high_traffic = np.sum(traffic_intensity_levels >= 0.3)
        
        print(f"Traffic patterns - Low: {low_traffic}, Medium: {med_traffic}, High: {high_traffic}")
        
        self.results['vehicle_time'] = {
            'traffic_intensity': traffic_intensity_levels,
            'closure_patterns': closure_patterns,
            'pattern_distribution': {'low': low_traffic, 'medium': med_traffic, 'high': high_traffic}
        }
        
        return traffic_intensity_levels, closure_patterns
    
    def analyze_path_characteristics(self):
        """Analyze optimal path diversity and complexity"""
        print("\n" + "="*60)
        print("5. OPTIMAL PATH CHARACTERISTICS ANALYSIS")
        print("="*60)
        
        path_lengths = []
        path_road_types = []
        path_costs = []
        
        for i, (input_tokens, label_tokens) in enumerate(zip(self.train_inputs, self.train_labels)):
            input_grid = self.tokens_to_grid(input_tokens)
            label_grid = self.tokens_to_grid(label_tokens)
            
            # Find path positions
            path_positions = np.where(label_grid == LOGISTICS_TOKEN_MAP['PATH'])
            
            if len(path_positions[0]) > 0:
                path_length = len(path_positions[0])
                path_lengths.append(path_length)
                
                # Analyze road types used in path
                road_types_in_path = []
                total_cost = 0
                
                for py, px in zip(path_positions[0], path_positions[1]):
                    original_road_type = input_grid[py, px]
                    road_types_in_path.append(original_road_type)
                    
                    # Calculate cost based on road type (from logistics game logic)
                    if original_road_type == LOGISTICS_TOKEN_MAP['TRAFFIC_JAM']:
                        total_cost += 8
                    elif original_road_type == LOGISTICS_TOKEN_MAP['LARGE_ROAD']:
                        total_cost += 1
                    elif original_road_type == LOGISTICS_TOKEN_MAP['BROADWAY']:
                        total_cost += 2
                    elif original_road_type == LOGISTICS_TOKEN_MAP['SMALL_ROAD']:
                        total_cost += 3
                    else:
                        total_cost += 1
                
                path_road_types.append(road_types_in_path)
                path_costs.append(total_cost)
        
        path_lengths = np.array(path_lengths)
        path_costs = np.array(path_costs)
        
        print(f"Path lengths - Min: {path_lengths.min()}, Max: {path_lengths.max()}, Mean: {path_lengths.mean():.1f}")
        print(f"Path length std: {path_lengths.std():.1f}")
        print(f"Path costs - Min: {path_costs.min()}, Max: {path_costs.max()}, Mean: {path_costs.mean():.1f}")
        print(f"Path cost std: {path_costs.std():.1f}")
        
        # Analyze road type usage in paths
        all_road_types = [road_type for path in path_road_types for road_type in path]
        road_type_counter = Counter(all_road_types)
        
        print("\nRoad type usage in optimal paths:")
        for token_id, count in road_type_counter.most_common():
            token_name = [k for k, v in LOGISTICS_TOKEN_MAP.items() if v == token_id][0] if token_id in LOGISTICS_TOKEN_MAP.values() else f"Unknown({token_id})"
            print(f"  {token_name}: {count} ({count/len(all_road_types)*100:.1f}%)")
        
        # Check for identical paths
        unique_paths = np.unique(self.train_labels, axis=0)
        path_duplicates = len(self.train_labels) - len(unique_paths)
        print(f"🚨 Identical paths: {path_duplicates} ({path_duplicates/len(self.train_labels)*100:.1f}%)")
        
        self.results['paths'] = {
            'lengths': path_lengths,
            'costs': path_costs,
            'road_type_usage': dict(road_type_counter),
            'identical_paths': path_duplicates/len(self.train_labels)
        }
        
        return path_lengths, path_costs
    
    def analyze_a_star_quality(self):
        """Validate A* solution quality"""
        print("\n" + "="*60)
        print("6. A* ORACLE SOLUTION QUALITY")
        print("="*60)
        
        valid_paths = 0
        invalid_paths = 0
        connectivity_issues = 0
        
        for i, (input_tokens, label_tokens) in enumerate(zip(self.train_inputs[:100], self.train_labels[:100])):  # Sample first 100
            input_grid = self.tokens_to_grid(input_tokens)
            label_grid = self.tokens_to_grid(label_tokens)
            
            # Find start, end, and path
            start_pos = np.where(input_grid == LOGISTICS_TOKEN_MAP['START'])
            end_pos = np.where(input_grid == LOGISTICS_TOKEN_MAP['END'])
            path_pos = np.where(label_grid == LOGISTICS_TOKEN_MAP['PATH'])
            
            if len(start_pos[0]) > 0 and len(end_pos[0]) > 0 and len(path_pos[0]) > 0:
                start = (start_pos[1][0], start_pos[0][0])  # (x, y)
                end = (end_pos[1][0], end_pos[0][0])
                path_positions = list(zip(path_pos[1], path_pos[0]))
                
                # Check if path connects start and end
                path_set = set(path_positions)
                
                # Check connectivity (path should form a connected sequence)
                if len(path_positions) > 1:
                    connected = True
                    # Simple connectivity check - each position should have at least one neighbor in path
                    for px, py in path_positions:
                        neighbors = [(px+1,py), (px-1,py), (px,py+1), (px,py-1)]
                        neighbor_in_path = any(n in path_set for n in neighbors)
                        if not neighbor_in_path and (px, py) != start and (px, py) != end:
                            connected = False
                            break
                    
                    if not connected:
                        connectivity_issues += 1
                
                if start in path_set and end in path_set:
                    valid_paths += 1
                else:
                    invalid_paths += 1
            else:
                invalid_paths += 1
        
        print(f"Valid paths: {valid_paths}")
        print(f"🚨 Invalid paths: {invalid_paths}")
        print(f"🚨 Connectivity issues: {connectivity_issues}")
        
        quality_score = valid_paths / (valid_paths + invalid_paths) if (valid_paths + invalid_paths) > 0 else 0
        print(f"A* solution quality: {quality_score:.2%}")
        
        self.results['a_star'] = {
            'valid_paths': valid_paths,
            'invalid_paths': invalid_paths,
            'connectivity_issues': connectivity_issues,
            'quality_score': quality_score
        }
    
    def generate_visualizations(self):
        """Generate comprehensive visualization plots"""
        print("\n" + "="*60)
        print("7. GENERATING VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('City Logistics Dataset Quality Analysis', fontsize=16)
        
        # Distance distribution
        if 'positions' in self.results:
            distances = [abs(s[0]-e[0]) + abs(s[1]-e[1]) for s, e in 
                        zip(self.results['positions']['start_positions'], 
                            self.results['positions']['end_positions'])]
            axes[0,0].hist(distances, bins=20, alpha=0.7, color='blue')
            axes[0,0].set_title('Start-End Manhattan Distances')
            axes[0,0].set_xlabel('Distance')
            axes[0,0].set_ylabel('Count')
        
        # Road type distribution
        if 'structure' in self.results:
            road_dist = self.results['structure']['road_distributions']
            if road_dist:
                small_roads = [rd['small_roads'] for rd in road_dist]
                large_roads = [rd['large_roads'] for rd in road_dist]
                broadway = [rd['broadway'] for rd in road_dist]
                
                axes[0,1].hist([small_roads, large_roads, broadway], 
                              bins=15, alpha=0.7, label=['Small Roads', 'Large Roads', 'Broadway'], 
                              color=['lightblue', 'darkblue', 'purple'])
                axes[0,1].set_title('Road Type Distribution')
                axes[0,1].set_xlabel('Count per Map')
                axes[0,1].set_ylabel('Frequency')
                axes[0,1].legend()
        
        # Traffic intensity distribution
        if 'vehicle_time' in self.results:
            traffic = self.results['vehicle_time']['traffic_intensity']
            axes[0,2].hist(traffic, bins=20, alpha=0.7, color='orange')
            axes[0,2].set_title('Traffic Intensity Distribution')
            axes[0,2].set_xlabel('Traffic Intensity Ratio')
            axes[0,2].set_ylabel('Count')
        
        # Path lengths
        if 'paths' in self.results:
            path_lengths = self.results['paths']['lengths']
            axes[1,0].hist(path_lengths, bins=20, alpha=0.7, color='green')
            axes[1,0].set_title('Optimal Path Lengths')
            axes[1,0].set_xlabel('Path Length')
            axes[1,0].set_ylabel('Count')
            
            # Path costs
            path_costs = self.results['paths']['costs']
            axes[1,1].hist(path_costs, bins=20, alpha=0.7, color='red')
            axes[1,1].set_title('Path Costs (Travel Time)')
            axes[1,1].set_xlabel('Cost')
            axes[1,1].set_ylabel('Count')
        
        # Uniqueness metrics
        if 'uniqueness' in self.results:
            uniqueness_metrics = ['Input\nDuplicates', 'Label\nDuplicates', 'Pair\nDuplicates', 'Train/Test\nLeakage']
            uniqueness_values = [
                self.results['uniqueness']['input_duplicates'],
                self.results['uniqueness']['label_duplicates'], 
                self.results['uniqueness']['pair_duplicates'],
                self.results['uniqueness']['train_test_leakage']
            ]
            
            bars = axes[1,2].bar(uniqueness_metrics, uniqueness_values, 
                               color=['red' if v > 0 else 'green' for v in uniqueness_values])
            axes[1,2].set_title('Data Quality Issues (0 = Good)')
            axes[1,2].set_ylabel('Count')
        
        # Sample map visualization
        if len(self.train_inputs) > 0:
            sample_input = self.tokens_to_grid(self.train_inputs[0])
            sample_label = self.tokens_to_grid(self.train_labels[0])
            
            # Input map
            im1 = axes[2,0].imshow(sample_input, cmap='tab10', vmin=0, vmax=9)
            axes[2,0].set_title('Sample Input Map')
            axes[2,0].set_xticks([])
            axes[2,0].set_yticks([])
            
            # Output map (path overlay)
            combined_map = sample_input.copy().astype(float)
            path_mask = sample_label == LOGISTICS_TOKEN_MAP['PATH']
            combined_map[path_mask] = LOGISTICS_TOKEN_MAP['PATH']
            
            im2 = axes[2,1].imshow(combined_map, cmap='tab10', vmin=0, vmax=9)
            axes[2,1].set_title('Sample Output (with Path)')
            axes[2,1].set_xticks([])
            axes[2,1].set_yticks([])
        
        # Road type usage in paths
        if 'paths' in self.results and 'road_type_usage' in self.results['paths']:
            road_usage = self.results['paths']['road_type_usage']
            token_names = []
            usage_counts = []
            
            for token_id, count in road_usage.items():
                token_name = [k for k, v in LOGISTICS_TOKEN_MAP.items() if v == token_id]
                if token_name:
                    token_names.append(token_name[0])
                    usage_counts.append(count)
            
            if token_names:
                axes[2,2].bar(token_names, usage_counts, color='skyblue')
                axes[2,2].set_title('Road Type Usage in Paths')
                axes[2,2].set_ylabel('Usage Count')
                axes[2,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('logistics_data_analysis.png', dpi=150, bbox_inches='tight')
        print("Saved comprehensive visualization to: logistics_data_analysis.png")
    
    def generate_report(self):
        """Generate comprehensive analysis report with specific logistics criteria"""
        print("\n" + "="*60)
        print("8. LOGISTICS DATASET QUALITY REPORT")
        print("="*60)
        
        severity_score = 0
        issues = []
        
        # Check uniqueness issues
        if 'uniqueness' in self.results:
            if self.results['uniqueness']['pair_duplicates'] > 0:
                severity_score += 10
                issues.append(f"🚨 CRITICAL: {self.results['uniqueness']['pair_duplicates']} exact duplicate examples")
            
            if self.results['uniqueness']['train_test_leakage'] > 0:
                severity_score += 10
                issues.append(f"🚨 CRITICAL: {self.results['uniqueness']['train_test_leakage']} examples in both train/test")
        
        # Check city road structure diversity
        if 'structure' in self.results:
            diversity_metrics = self.results['structure']['diversity_metrics']
            
            if diversity_metrics['obstacle_std'] < 0.05:
                severity_score += 6
                issues.append(f"🚨 MEDIUM: Low obstacle diversity (std: {diversity_metrics['obstacle_std']:.3f})")
            
            if diversity_metrics['small_road_std'] < 0.05:
                severity_score += 4
                issues.append(f"🚨 LOW: Low small road diversity (std: {diversity_metrics['small_road_std']:.3f})")
            
            if diversity_metrics['broadway_std'] < 0.02:
                severity_score += 3
                issues.append(f"🚨 LOW: Low Broadway diversity (std: {diversity_metrics['broadway_std']:.3f})")
        
        # Check traffic/time diversity
        if 'vehicle_time' in self.results:
            traffic_std = self.results['vehicle_time']['traffic_intensity'].std()
            if traffic_std < 0.05:
                severity_score += 5
                issues.append(f"🚨 MEDIUM: Low traffic pattern diversity (std: {traffic_std:.3f})")
        
        # Check path diversity
        if 'paths' in self.results:
            if self.results['paths']['identical_paths'] > 0.1:
                severity_score += 7
                issues.append(f"🚨 HIGH: {self.results['paths']['identical_paths']*100:.1f}% identical paths")
        
        # Check A* solution quality
        if 'a_star' in self.results:
            quality_score = self.results['a_star']['quality_score']
            if quality_score < 0.95:
                severity_score += 8
                issues.append(f"🚨 HIGH: A* solution quality only {quality_score:.1%}")
        
        print(f"SEVERITY SCORE: {severity_score}/50 (higher = worse)")
        
        if severity_score <= 10:
            recommendation = "✅ EXCELLENT - Ready for HRM training"
        elif severity_score <= 20:
            recommendation = "✅ GOOD - Minor improvements suggested"
        elif severity_score <= 30:
            recommendation = "⚠️ ACCEPTABLE - Some improvements needed"
        else:
            recommendation = "🚨 POOR - Significant improvements required"
        
        print(f"RECOMMENDATION: {recommendation}")
        
        print("\nISSUES FOUND:")
        if not issues:
            print("  ✅ No significant issues detected!")
        else:
            for issue in issues:
                print(f"  {issue}")
        
        # City-specific recommendations
        print("\n🗽 CITY LOGISTICS SPECIFIC RECOMMENDATIONS:")
        print("  1. Ensure vehicle type diversity (bike/car/van/truck restrictions)")
        print("  2. Verify time-based traffic patterns (rush hour vs off-peak)")
        print("  3. Check road hierarchy (small roads < large roads < Broadway)")
        print("  4. Validate A* oracle produces optimal routes")
        print("  5. Confirm traffic conditions affect path costs correctly")
        
        return severity_score, issues
    
    def run_full_analysis(self):
        """Run complete logistics dataset quality analysis"""
        print("Starting comprehensive city logistics dataset analysis...")
        print(f"Dataset location: {self.data_dir}")
        
        try:
            # Run all analyses
            self.analyze_uniqueness()
            distances, start_pos, end_pos = self.analyze_start_end_positions()
            obstacle_ratios, road_distributions = self.analyze_road_structure()
            traffic_intensity, closure_patterns = self.analyze_vehicle_time_diversity()
            path_lengths, path_costs = self.analyze_path_characteristics()
            self.analyze_a_star_quality()
            
            # Generate visualizations
            self.generate_visualizations()
            
            # Generate final report
            severity_score, issues = self.generate_report()
            
            return severity_score < 25  # Return True if dataset is acceptable
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    analyzer = LogisticsDataAnalyzer()
    is_acceptable = analyzer.run_full_analysis()
    
    if is_acceptable:
        print("\n" + "="*60)
        print("CONCLUSION: Dataset quality is suitable for HRM training!")
        print("Proceed with training using the converted .npy files.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("CONCLUSION: Dataset quality issues detected.")
        print("Consider regenerating data with improvements.")
        print("="*60)