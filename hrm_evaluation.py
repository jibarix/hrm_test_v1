#!/usr/bin/env python3
"""
Fixed HRM Evaluation - All token assignment issues resolved
"""

import requests
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import time
import random

class ProperHRMEvaluator:
    def __init__(self, server_url: str = "http://127.0.0.1:5000"):
        self.server_url = server_url
        self.map_width = 30
        self.map_height = 30
        
        # Exact token mapping from your codebase
        self.HRM_TOKEN_MAP = {
            "PAD": 0,
            "OBSTACLE": 1,      # Buildings/City Park
            "SMALL_ROAD": 2,    # Side Streets  
            "LARGE_ROAD": 3,    # Major Avenues
            "DIAGONAL": 4,      # Main diagonal thoroughfare
            "TRAFFIC_JAM": 5,   # Heavy Traffic
            "ROAD_CLOSURE": 6,  # Road Closure
            "START": 7,         # Start Point
            "END": 8,           # End Point
            "PATH": 9           # Optimal Route
        }
        
        # Test connection
        self.test_connection()
    
    def test_connection(self):
        """Test if the HRM server is running"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"Connected to HRM server at {self.server_url}")
            else:
                print(f"Server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Cannot connect to server: {e}")
            raise
    
    def generate_city_logistics_scenario(self) -> Tuple[List[int], List[int], Dict]:
        """Generate scenario using exact logic from logistics_game.html"""
        
        # Create base grid using the same logic as logistics_game.html
        grid = np.full((self.map_height, self.map_width), self.HRM_TOKEN_MAP["OBSTACLE"], dtype=int)
        
        # Add road network exactly like the logistics game
        # Major avenues (from logistics_game.html)
        avenues = [3, 8, 13, 18, 23, 28, 33, 37]
        for avenue in avenues:
            for y in range(self.map_height):
                if avenue < self.map_width:
                    grid[y][avenue] = self.HRM_TOKEN_MAP["LARGE_ROAD"]
                    if avenue + 1 < self.map_width:
                        grid[y][avenue + 1] = self.HRM_TOKEN_MAP["LARGE_ROAD"]

        # Major streets (from logistics_game.html) 
        streets = [2, 6, 10, 14, 18, 22, 26, 30, 34, 37]
        for street in streets:
            for x in range(self.map_width):
                if street < self.map_height:
                    grid[street][x] = self.HRM_TOKEN_MAP["LARGE_ROAD"]

        # Side streets
        for y in range(4, self.map_height, 4):
            for x in range(self.map_width):
                if grid[y][x] == self.HRM_TOKEN_MAP["OBSTACLE"]:
                    grid[y][x] = self.HRM_TOKEN_MAP["SMALL_ROAD"]

        for x in range(5, self.map_width, 6):
            for y in range(self.map_height):
                if grid[y][x] == self.HRM_TOKEN_MAP["OBSTACLE"]:
                    grid[y][x] = self.HRM_TOKEN_MAP["SMALL_ROAD"]

        # Add main diagonal (from logistics_game.html)
        for i in range(self.map_width - 5):
            x = i
            y = int((i * self.map_height) / self.map_width)
            if y < self.map_height and x < self.map_width:
                grid[y][x] = self.HRM_TOKEN_MAP["DIAGONAL"]
                if y + 1 < self.map_height:
                    grid[y + 1][x] = self.HRM_TOKEN_MAP["DIAGONAL"]

        # Add central park (from logistics_game.html)
        park_start_x = int(self.map_width * 0.4)
        park_end_x = int(self.map_width * 0.65)
        park_start_y = int(self.map_height * 0.15)
        park_end_y = int(self.map_height * 0.45)
        
        for y in range(park_start_y, park_end_y):
            for x in range(park_start_x, park_end_x):
                if x < self.map_width and y < self.map_height:
                    grid[y][x] = self.HRM_TOKEN_MAP["OBSTACLE"]
        
        # Add traffic using the fixed logic
        current_hour = random.choice([6, 8, 10, 12, 14, 17, 20, 23])
        traffic_grid = self.add_traffic_fixed(grid, current_hour)
        
        # Find valid start and end positions (on roads, not obstacles)
        road_positions = []
        for y in range(self.map_height):
            for x in range(self.map_width):
                if traffic_grid[y][x] in [self.HRM_TOKEN_MAP["SMALL_ROAD"], 
                                         self.HRM_TOKEN_MAP["LARGE_ROAD"], 
                                         self.HRM_TOKEN_MAP["DIAGONAL"],
                                         self.HRM_TOKEN_MAP["TRAFFIC_JAM"]]:
                    # Check edge distance like in the original code
                    edge_distance = min(x, y, self.map_width - 1 - x, self.map_height - 1 - y)
                    if edge_distance >= 2:
                        road_positions.append((x, y))
        
        if len(road_positions) < 2:
            print("Warning: Not enough road positions found!")
            return self.generate_city_logistics_scenario()  # Try again
        
        # Select start and end with good distance (like placeStartAndEnd)
        attempts = 0
        while attempts < 10:
            start_idx = random.randint(0, len(road_positions) - 1)
            start_pos = road_positions[start_idx]
            
            valid_end_positions = []
            for pos in road_positions:
                distance = abs(pos[0] - start_pos[0]) + abs(pos[1] - start_pos[1])
                if 15 <= distance <= 50:
                    valid_end_positions.append(pos)
            
            if valid_end_positions:
                end_pos = random.choice(valid_end_positions)
                break
            attempts += 1
        else:
            # Fallback
            start_pos = road_positions[0]
            end_pos = road_positions[-1]
        
        # Create input sequence using exact gridToHRMSequence logic
        input_sequence = []
        for y in range(self.map_height):
            for x in range(self.map_width):
                if start_pos and start_pos[0] == x and start_pos[1] == y:
                    input_sequence.append(self.HRM_TOKEN_MAP["START"])
                elif end_pos and end_pos[0] == x and end_pos[1] == y:
                    input_sequence.append(self.HRM_TOKEN_MAP["END"])
                else:
                    input_sequence.append(int(traffic_grid[y][x]))
        
        # Generate oracle path using A* (like your evaluation)
        oracle_path = self.solve_path_astar(traffic_grid, start_pos, end_pos)
        
        # Create oracle sequence using pathToHRMSequence logic
        oracle_sequence = input_sequence.copy()
        for pos in oracle_path:
            idx = pos[1] * self.map_width + pos[0]  # y * width + x
            if oracle_sequence[idx] not in [self.HRM_TOKEN_MAP["START"], self.HRM_TOKEN_MAP["END"]]:
                oracle_sequence[idx] = self.HRM_TOKEN_MAP["PATH"]
        
        metadata = {
            'start': start_pos,
            'end': end_pos,
            'oracle_path': oracle_path,
            'oracle_length': len(oracle_path),
            'hour': current_hour
        }
        
        return input_sequence, oracle_sequence, metadata
    
    def add_traffic_fixed(self, base_grid, hour):
        """Add traffic with all token assignments as integers"""
        traffic_grid = base_grid.copy()
        
        # Traffic intensity logic from addTraffic
        construction_active = False
        if 7 <= hour <= 9:
            traffic_intensity = 0.4
        elif 17 <= hour <= 19:
            traffic_intensity = 0.45
        elif 12 <= hour <= 13:
            traffic_intensity = 0.25
        elif 2 <= hour <= 5:
            traffic_intensity = 0.1
            construction_active = True
        else:
            traffic_intensity = 0.1

        for y in range(self.map_height):
            for x in range(self.map_width):
                road_type = base_grid[y][x]
                
                if road_type in [self.HRM_TOKEN_MAP["LARGE_ROAD"], self.HRM_TOKEN_MAP["DIAGONAL"]]:
                    if construction_active and random.random() < 0.05:
                        traffic_grid[y][x] = self.HRM_TOKEN_MAP["ROAD_CLOSURE"]  # Integer 6
                    elif random.random() < traffic_intensity:
                        traffic_grid[y][x] = self.HRM_TOKEN_MAP["TRAFFIC_JAM"]   # Integer 5
                elif road_type == self.HRM_TOKEN_MAP["SMALL_ROAD"]:
                    if random.random() < traffic_intensity * 0.3:
                        traffic_grid[y][x] = self.HRM_TOKEN_MAP["TRAFFIC_JAM"]   # Integer 5
        
        return traffic_grid
    
    def solve_path_astar(self, dynamic_grid, start, end):
        """A* pathfinding using the same vehicle constraints as your evaluation"""
        from heapq import heappush, heappop
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(node):
            neighbors = []
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in dirs:
                next_x = node[0] + dx
                next_y = node[1] + dy
                if 0 <= next_x < self.map_width and 0 <= next_y < self.map_height:
                    tile_type = dynamic_grid[next_y][next_x]
                    
                    # Can traverse roads and traffic, but not obstacles or closures
                    if (tile_type != self.HRM_TOKEN_MAP["OBSTACLE"] and 
                        tile_type != self.HRM_TOKEN_MAP["ROAD_CLOSURE"] and
                        tile_type in [self.HRM_TOKEN_MAP["SMALL_ROAD"], 
                                     self.HRM_TOKEN_MAP["LARGE_ROAD"],
                                     self.HRM_TOKEN_MAP["DIAGONAL"], 
                                     self.HRM_TOKEN_MAP["TRAFFIC_JAM"]]):
                        neighbors.append((next_x, next_y))
            return neighbors
        
        def get_cost(tile_type):
            if tile_type == self.HRM_TOKEN_MAP["TRAFFIC_JAM"]:
                return 8
            elif tile_type == self.HRM_TOKEN_MAP["ROAD_CLOSURE"]:
                return float('inf')
            elif tile_type == self.HRM_TOKEN_MAP["LARGE_ROAD"]:
                return 1
            elif tile_type == self.HRM_TOKEN_MAP["DIAGONAL"]:
                return 2
            elif tile_type == self.HRM_TOKEN_MAP["SMALL_ROAD"]:
                return 3
            else:
                return 1
        
        open_set = [(0, start)]
        came_from = {}
        closed_set = set()
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        while open_set:
            current_f, current = heappop(open_set)
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]  # Reverse to get start->end order
            
            closed_set.add(current)
            
            for neighbor in get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                cost = get_cost(dynamic_grid[neighbor[1]][neighbor[0]])
                tentative_g_score = g_score[current] + cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def check_start_end_as_path(self, predicted_seq: List[int], start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> Tuple[bool, bool]:
        """Check if model outputs PATH tokens at start/end positions (the correct behavior)"""
        start_idx = start_pos[1] * self.map_width + start_pos[0] 
        end_idx = end_pos[1] * self.map_width + end_pos[0]
        
        start_has_path = predicted_seq[start_idx] == self.HRM_TOKEN_MAP["PATH"]
        end_has_path = predicted_seq[end_idx] == self.HRM_TOKEN_MAP["PATH"]
        
        return start_has_path, end_has_path

    def extract_path_coords_from_logits(self, predicted_sequence: List[int]) -> List[Tuple[int, int]]:
        """Extract PATH coordinates from model prediction using app.py logic"""
        path_coords = []
        
        for i, token in enumerate(predicted_sequence):
            if token == self.HRM_TOKEN_MAP["PATH"]:
                y = i // self.map_width
                x = i % self.map_width
                path_coords.append((x, y))
        
        return path_coords
    
    def extract_start_end_from_sequence(self, sequence: List[int]) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Extract start and end positions from sequence"""
        start_pos = None
        end_pos = None
        
        for i, token in enumerate(sequence):
            y = i // self.map_width
            x = i % self.map_width
            
            if token == self.HRM_TOKEN_MAP["START"]:
                start_pos = (x, y)
            elif token == self.HRM_TOKEN_MAP["END"]:
                end_pos = (x, y)
        
        return start_pos, end_pos
    
    def get_model_prediction(self, input_sequence: List[int]) -> Optional[List[int]]:
        """Get prediction from HRM server"""
        try:
            payload = {
                "inputs": input_sequence,
                "log_intermediate": True
            }
            
            response = requests.post(
                f"{self.server_url}/predict", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['path']  # This is the predicted sequence
            else:
                print(f"Server error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    def evaluate_prediction(self, input_seq: List[int], predicted_seq: List[int], oracle_seq: List[int], metadata: Dict) -> Dict:
        """Evaluate a single prediction"""
        
        # Extract positions using exact same logic as the model
        pred_start, pred_end = self.extract_start_end_from_sequence(predicted_seq)
        oracle_start, oracle_end = self.extract_start_end_from_sequence(oracle_seq)
        
        # NEW: Check if model outputs PATH tokens at start/end positions (correct behavior)
        start_as_path, end_as_path = self.check_start_end_as_path(predicted_seq, metadata['start'], metadata['end'])
        
        pred_path = self.extract_path_coords_from_logits(predicted_seq)
        oracle_path = metadata['oracle_path']
        
        # Exact match
        exact_match = predicted_seq == oracle_seq
        
        # Start/end accuracy (original method - looking for START/END tokens)
        start_correct = pred_start == oracle_start
        end_correct = pred_end == oracle_end
        
        # NEW: Start/end accuracy (correct method - model should output PATH at these positions)
        start_correct_as_path = start_as_path  # Model should put PATH at start
        end_correct_as_path = end_as_path      # Model should put PATH at end
        
        # Path overlap
        pred_path_set = set(pred_path)
        oracle_path_set = set(oracle_path)
        
        if pred_path:
            path_overlap = len(pred_path_set.intersection(oracle_path_set)) / len(pred_path_set)
        else:
            path_overlap = 0.0
        
        # Path length comparison
        path_length_ratio = len(pred_path) / max(len(oracle_path), 1)
        path_efficiency = len(oracle_path) / max(len(pred_path), 1)
        
        # Validity checks
        valid_path = self.is_valid_path(input_seq, pred_path)
        connected_path = self.is_connected_path(pred_path)
        optimal_path = (len(pred_path) == len(oracle_path) and valid_path and connected_path)
        
        return {
            'exact_match': exact_match,
            'start_correct': start_correct,
            'end_correct': end_correct,
            'start_correct_as_path': start_correct_as_path,  # NEW
            'end_correct_as_path': end_correct_as_path,      # NEW
            'path_overlap': path_overlap,
            'path_length_ratio': path_length_ratio,
            'path_efficiency': path_efficiency,
            'valid_path': valid_path,
            'connected_path': connected_path,
            'optimal_path': optimal_path,
            'pred_path_length': len(pred_path),
            'oracle_path_length': len(oracle_path)
        }
    
    def is_valid_path(self, input_sequence: List[int], path_coords: List[Tuple[int, int]]) -> bool:
        """Check if path goes through valid tiles"""
        for x, y in path_coords:
            idx = y * self.map_width + x
            if idx >= len(input_sequence):
                return False
            
            token = input_sequence[idx]
            # Path should not go through obstacles or road closures
            if token in [self.HRM_TOKEN_MAP["OBSTACLE"], self.HRM_TOKEN_MAP["ROAD_CLOSURE"]]:
                return False
        return True
    
    def is_connected_path(self, path_coords: List[Tuple[int, int]]) -> bool:
        """Check if path forms a connected sequence"""
        if len(path_coords) <= 1:
            return True
        
        coord_set = set(path_coords)
        for x, y in path_coords:
            adjacent = False
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if (x + dx, y + dy) in coord_set:
                    adjacent = True
                    break
            if not adjacent and len(path_coords) > 1:
                return False
        return True
    
    def run_proper_evaluation(self, num_tests: int = 10) -> Dict:
        """Run proper evaluation with correct data format"""
        print(f"Running {num_tests} properly formatted tests...")
        
        all_results = []
        inference_times = []
        
        for i in range(num_tests):
            print(f"Test {i+1}/{num_tests}...", end=" ")
            
            try:
                # Generate proper scenario
                input_seq, oracle_seq, metadata = self.generate_city_logistics_scenario()
                
                # Get model prediction
                start_time = time.time()
                predicted_seq = self.get_model_prediction(input_seq)
                inference_time = time.time() - start_time
                
                if predicted_seq is None:
                    print("FAILED (no prediction)")
                    continue
                
                # Evaluate
                result = self.evaluate_prediction(input_seq, predicted_seq, oracle_seq, metadata)
                all_results.append(result)
                inference_times.append(inference_time)
                
                print(f"OK ({inference_time:.2f}s, path: {result['pred_path_length']} vs {result['oracle_path_length']} oracle)")
            
            except Exception as e:
                print(f"FAILED ({e})")
                continue
        
        if not all_results:
            return {}
        
        # Aggregate results
        aggregated = {
            'exact_accuracy': np.mean([r['exact_match'] for r in all_results]),
            'start_point_accuracy': np.mean([r['start_correct'] for r in all_results]),
            'end_point_accuracy': np.mean([r['end_correct'] for r in all_results]),
            'start_point_accuracy_as_path': np.mean([r['start_correct_as_path'] for r in all_results]),  # NEW
            'end_point_accuracy_as_path': np.mean([r['end_correct_as_path'] for r in all_results]),      # NEW
            'avg_path_overlap': np.mean([r['path_overlap'] for r in all_results]),
            'avg_path_length_ratio': np.mean([r['path_length_ratio'] for r in all_results]),
            'avg_path_efficiency': np.mean([r['path_efficiency'] for r in all_results]),
            'valid_path_rate': np.mean([r['valid_path'] for r in all_results]),
            'connected_path_rate': np.mean([r['connected_path'] for r in all_results]),
            'optimal_path_rate': np.mean([r['optimal_path'] for r in all_results]),
            'avg_inference_time': np.mean(inference_times),
            'avg_predicted_path_length': np.mean([r['pred_path_length'] for r in all_results]),
            'avg_oracle_path_length': np.mean([r['oracle_path_length'] for r in all_results]),
            'total_tests': len(all_results)
        }
        
        return aggregated


def main():
    print("Fixed HRM Evaluation Using Exact Codebase Logic")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ProperHRMEvaluator()
    
    # Run tests
    num_tests = int(input("How many test scenarios? (default: 10): ") or "10")
    
    results = evaluator.run_proper_evaluation(num_tests)
    
    if not results:
        print("No successful tests to analyze!")
        return
    
    # Display results
    print("\n" + "=" * 60)
    print("PROPER EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"Total successful tests: {results['total_tests']}")
    print(f"Average inference time: {results['avg_inference_time']:.3f}s")
    print()
    
    print("ACCURACY METRICS:")
    print(f"  Exact Accuracy: {results['exact_accuracy']:.1%}")
    print(f"  Start Point Accuracy (looking for START token): {results['start_point_accuracy']:.1%}")  
    print(f"  End Point Accuracy (looking for END token): {results['end_point_accuracy']:.1%}")
    print(f"  Start Point Accuracy (as PATH token): {results['start_point_accuracy_as_path']:.1%}")  # NEW
    print(f"  End Point Accuracy (as PATH token): {results['end_point_accuracy_as_path']:.1%}")      # NEW
    print()
    
    print("PATH QUALITY METRICS:")
    print(f"  Path Overlap: {results['avg_path_overlap']:.1%}")
    print(f"  Path Length Ratio: {results['avg_path_length_ratio']:.3f}")
    print(f"  Path Efficiency: {results['avg_path_efficiency']:.3f}")
    print()
    
    print("VALIDITY METRICS:")
    print(f"  Valid Path Rate: {results['valid_path_rate']:.1%}")
    print(f"  Connected Path Rate: {results['connected_path_rate']:.1%}")  
    print(f"  Optimal Path Rate: {results['optimal_path_rate']:.1%}")
    print()
    
    print("PATH LENGTH ANALYSIS:")
    print(f"  Avg Predicted Length: {results['avg_predicted_path_length']:.1f}")
    print(f"  Avg Oracle Length: {results['avg_oracle_path_length']:.1f}")
    print()
    
    # Analysis
    print("ANALYSIS:")
    print("-" * 30)
    
    if results['start_point_accuracy_as_path'] > 0.8:
        print("✓ Model correctly identifies start positions (via PATH tokens)")
    elif results['start_point_accuracy'] > 0.8:
        print("✓ Model correctly identifies start positions (via START tokens)")
    else:
        print("✗ Model struggles with start position detection")
    
    if results['end_point_accuracy_as_path'] > 0.8:
        print("✓ Model correctly identifies end positions (via PATH tokens)")
    elif results['end_point_accuracy'] > 0.8:
        print("✓ Model correctly identifies end positions (via END tokens)")
    else:
        print("✗ Model struggles with end position detection")
    
    # Show the format discovery
    if results['start_point_accuracy_as_path'] > results['start_point_accuracy']:
        print(f"🔍 DISCOVERY: Model uses PATH tokens for start/end (correct training format!)")
        print(f"   Start as PATH: {results['start_point_accuracy_as_path']:.1%} vs START token: {results['start_point_accuracy']:.1%}")
        print(f"   End as PATH: {results['end_point_accuracy_as_path']:.1%} vs END token: {results['end_point_accuracy']:.1%}")
    
    if results['valid_path_rate'] > 0.9:
        print("✓ Model generates valid paths (avoids obstacles)")
    else:
        print("✗ Model sometimes generates invalid paths")
    
    if results['avg_path_overlap'] > 0.5:
        print("✓ Model paths have good overlap with optimal routes")
    else:
        print("~ Model paths deviate from optimal routes")
    
    if 0.8 <= results['avg_path_length_ratio'] <= 1.5:
        print("✓ Model generates reasonably sized paths")
    else:
        print(f"~ Model paths are {'too long' if results['avg_path_length_ratio'] > 1.5 else 'too short'}")
    
    if results['connected_path_rate'] > 0.8:
        print("✓ Model generates connected path sequences")
    else:
        print("✗ Model generates disconnected path fragments")
    
    # Save results
    output_dir = Path("proper_evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump({k: float(v) if isinstance(v, np.floating) else v for k, v in results.items()}, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()