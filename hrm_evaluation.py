#!/usr/bin/env python3
"""
FIXED HRM Evaluation - Proper pathfinding validation
Replaces token-counting with actual graph traversal and connectivity validation
"""

import requests
import json
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import matplotlib.pyplot as plt
from pathlib import Path
import time
import random
from collections import deque

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
            'hour': current_hour,
            'traffic_grid': traffic_grid.tolist()  # For validation
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

    # FIXED: Proper pathfinding validation methods
    
    def extract_path_coords_from_logits(self, predicted_sequence: List[int]) -> List[Tuple[int, int]]:
        """Extract PATH coordinates from model prediction"""
        path_coords = []
        
        for i, token in enumerate(predicted_sequence):
            if token == self.HRM_TOKEN_MAP["PATH"]:
                y = i // self.map_width
                x = i % self.map_width
                path_coords.append((x, y))
        
        return path_coords
    
    def trace_complete_path(self, predicted_sequence: List[int], start_pos: Tuple[int, int], end_pos: Tuple[int, int], traffic_grid: List[List[int]]) -> Dict:
        """
        FIXED: Trace a complete, connected path from start to end using BFS
        This replaces the broken token-counting approach
        """
        # Extract all PATH coordinates
        path_coords = self.extract_path_coords_from_logits(predicted_sequence)
        path_set = set(path_coords)
        
        # Add start and end to path set (they should be PATH tokens)
        if start_pos in path_set and end_pos in path_set:
            has_start_end_markers = True
        else:
            has_start_end_markers = False
            # For tracing purposes, include start/end even if not marked as PATH
            path_set.add(start_pos)
            path_set.add(end_pos)
        
        if len(path_coords) == 0:
            return {
                'found_complete_path': False,
                'path_length': 0,
                'valid_traversal': False,
                'has_start_end_markers': has_start_end_markers,
                'traced_path': [],
                'reason': 'No PATH tokens found'
            }
        
        # BFS to trace from start to end using only PATH coordinates
        queue = deque([(start_pos, [start_pos])])
        visited = {start_pos}
        
        while queue:
            current_pos, current_path = queue.popleft()
            
            if current_pos == end_pos:
                # Found complete path!
                return {
                    'found_complete_path': True,
                    'path_length': len(current_path),
                    'valid_traversal': self.is_path_traversable(current_path, traffic_grid),
                    'has_start_end_markers': has_start_end_markers,
                    'traced_path': current_path,
                    'reason': 'Complete path found'
                }
            
            # Check all adjacent cells
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_x = current_pos[0] + dx
                next_y = current_pos[1] + dy
                next_pos = (next_x, next_y)
                
                # Must be in bounds, be a PATH coordinate, and not visited
                if (0 <= next_x < self.map_width and 
                    0 <= next_y < self.map_height and
                    next_pos in path_set and 
                    next_pos not in visited):
                    
                    visited.add(next_pos)
                    queue.append((next_pos, current_path + [next_pos]))
        
        return {
            'found_complete_path': False,
            'path_length': len(path_coords),
            'valid_traversal': False,
            'has_start_end_markers': has_start_end_markers,
            'traced_path': [],
            'reason': f'No connected path from start to end (found {len(path_coords)} scattered PATH tokens)'
        }
    
    def is_path_traversable(self, path: List[Tuple[int, int]], traffic_grid: List[List[int]]) -> bool:
        """
        Check if the traced path goes through legal terrain
        """
        for x, y in path:
            if y >= len(traffic_grid) or x >= len(traffic_grid[0]):
                return False
            
            tile_type = traffic_grid[y][x]
            
            # Cannot traverse obstacles or road closures
            if tile_type in [self.HRM_TOKEN_MAP["OBSTACLE"], self.HRM_TOKEN_MAP["ROAD_CLOSURE"]]:
                return False
        
        return True
    
    def check_start_end_as_path(self, predicted_seq: List[int], start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> Tuple[bool, bool]:
        """Check if model outputs PATH tokens at start/end positions"""
        start_idx = start_pos[1] * self.map_width + start_pos[0] 
        end_idx = end_pos[1] * self.map_width + end_pos[0]
        
        start_has_path = predicted_seq[start_idx] == self.HRM_TOKEN_MAP["PATH"]
        end_has_path = predicted_seq[end_idx] == self.HRM_TOKEN_MAP["PATH"]
        
        return start_has_path, end_has_path
    
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
        """
        FIXED: Honest evaluation using proper pathfinding validation
        """
        
        start_pos = metadata['start']
        end_pos = metadata['end']
        oracle_path = metadata['oracle_path']
        traffic_grid = metadata['traffic_grid']
        
        # 1. Trace the actual path the model found
        path_analysis = self.trace_complete_path(predicted_seq, start_pos, end_pos, traffic_grid)
        
        # 2. Check start/end marker placement (this part was actually reasonable)
        start_as_path, end_as_path = self.check_start_end_as_path(predicted_seq, start_pos, end_pos)
        
        # 3. Extract all PATH coordinates for analysis
        pred_path_coords = self.extract_path_coords_from_logits(predicted_seq)
        
        # 4. Calculate meaningful metrics
        if path_analysis['found_complete_path'] and len(oracle_path) > 0:
            # True path overlap (only if we found a complete path)
            pred_path_set = set(path_analysis['traced_path'])
            oracle_path_set = set(oracle_path)
            overlap = len(pred_path_set.intersection(oracle_path_set)) / len(pred_path_set) if pred_path_set else 0
            
            # Path efficiency
            efficiency = len(oracle_path) / len(path_analysis['traced_path'])
            
            # Is it optimal?
            optimal_path = (path_analysis['path_length'] == len(oracle_path) and 
                          path_analysis['valid_traversal'] and 
                          path_analysis['found_complete_path'])
        else:
            # No complete path found
            overlap = 0.0
            efficiency = 0.0
            optimal_path = False
        
        # 5. Exact sequence match (usually meaningless, but included for completeness)
        exact_match = predicted_seq == oracle_seq
        
        return {
            # Core pathfinding metrics
            'found_complete_path': path_analysis['found_complete_path'],
            'valid_traversal': path_analysis['valid_traversal'],
            'path_analysis_reason': path_analysis['reason'],
            
            # Start/end detection
            'start_marked_as_path': start_as_path,
            'end_marked_as_path': end_as_path,
            
            # Path quality (only meaningful if complete path found)
            'path_overlap': overlap,
            'path_efficiency': efficiency,
            'optimal_path': optimal_path,
            
            # Length comparisons
            'predicted_path_length': path_analysis['path_length'],
            'oracle_path_length': len(oracle_path),
            'total_path_tokens': len(pred_path_coords),  # All PATH tokens, connected or not
            
            # Legacy metrics
            'exact_match': exact_match,
        }
    
    def run_proper_evaluation(self, num_tests: int = 10) -> Dict:
        """Run proper evaluation with honest pathfinding validation"""
        print(f"Running {num_tests} tests with proper pathfinding validation...")
        
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
                
                # FIXED: Honest evaluation
                result = self.evaluate_prediction(input_seq, predicted_seq, oracle_seq, metadata)
                all_results.append(result)
                inference_times.append(inference_time)
                
                # Honest reporting
                if result['found_complete_path']:
                    status = f"COMPLETE PATH ({result['predicted_path_length']} vs {result['oracle_path_length']} oracle)"
                else:
                    status = f"NO COMPLETE PATH ({result['total_path_tokens']} scattered tokens) - {result['path_analysis_reason']}"
                
                print(f"{status} ({inference_time:.2f}s)")
            
            except Exception as e:
                print(f"FAILED ({e})")
                continue
        
        if not all_results:
            return {}
        
        # Aggregate results with honest metrics
        aggregated = {
            # Core success metrics
            'complete_path_rate': np.mean([r['found_complete_path'] for r in all_results]),
            'valid_traversal_rate': np.mean([r['valid_traversal'] for r in all_results]),
            'optimal_path_rate': np.mean([r['optimal_path'] for r in all_results]),
            
            # Start/end detection (this part was actually reasonable)
            'start_detection_rate': np.mean([r['start_marked_as_path'] for r in all_results]),
            'end_detection_rate': np.mean([r['end_marked_as_path'] for r in all_results]),
            
            # Path quality (only for successful cases)
            'avg_path_overlap': np.mean([r['path_overlap'] for r in all_results if r['found_complete_path']]),
            'avg_path_efficiency': np.mean([r['path_efficiency'] for r in all_results if r['found_complete_path']]),
            
            # Length analysis
            'avg_predicted_path_length': np.mean([r['predicted_path_length'] for r in all_results]),
            'avg_oracle_path_length': np.mean([r['oracle_path_length'] for r in all_results]),
            'avg_total_path_tokens': np.mean([r['total_path_tokens'] for r in all_results]),
            
            # Meta
            'avg_inference_time': np.mean(inference_times),
            'total_tests': len(all_results),
            'successful_complete_paths': sum([r['found_complete_path'] for r in all_results]),
        }
        
        return aggregated


def main():
    print("FIXED HRM Evaluation - Proper Pathfinding Validation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ProperHRMEvaluator()
    
    # Run tests
    num_tests = int(input("How many test scenarios? (default: 10): ") or "10")
    
    results = evaluator.run_proper_evaluation(num_tests)
    
    if not results:
        print("No successful tests to analyze!")
        return
    
    # Display honest results
    print("\n" + "=" * 60)
    print("HONEST EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"Total successful tests: {results['total_tests']}")
    print(f"Average inference time: {results['avg_inference_time']:.3f}s")
    print()
    
    print("CORE PATHFINDING METRICS:")
    print(f"  Complete Path Rate: {results['complete_path_rate']:.1%}")
    print(f"  Valid Traversal Rate: {results['valid_traversal_rate']:.1%}")
    print(f"  Optimal Path Rate: {results['optimal_path_rate']:.1%}")
    print(f"  Successful Complete Paths: {results['successful_complete_paths']}/{results['total_tests']}")
    print()
    
    print("START/END DETECTION:")
    print(f"  Start Detection Rate: {results['start_detection_rate']:.1%}")  
    print(f"  End Detection Rate: {results['end_detection_rate']:.1%}")
    print()
    
    if results['successful_complete_paths'] > 0:
        print("PATH QUALITY (for successful paths only):")
        print(f"  Average Path Overlap: {results['avg_path_overlap']:.1%}")
        print(f"  Average Path Efficiency: {results['avg_path_efficiency']:.3f}")
    else:
        print("PATH QUALITY: N/A (no complete paths found)")
    print()
    
    print("PATH LENGTH ANALYSIS:")
    print(f"  Avg Complete Path Length: {results['avg_predicted_path_length']:.1f}")
    print(f"  Avg Oracle Length: {results['avg_oracle_path_length']:.1f}")
    print(f"  Avg Total PATH Tokens: {results['avg_total_path_tokens']:.1f}")
    print()
    
    # Honest analysis
    print("HONEST ANALYSIS:")
    print("-" * 30)
    
    if results['complete_path_rate'] > 0.8:
        print("SUCCESS: Model can find complete, connected paths")
    elif results['complete_path_rate'] > 0.3:
        print("PARTIAL: Model sometimes finds complete paths")
    else:
        print("FAILURE: Model rarely/never finds complete paths")
    
    if results['start_detection_rate'] > 0.8 and results['end_detection_rate'] > 0.8:
        print("SUCCESS: Model correctly marks start/end positions")
    else:
        print("ISSUE: Model struggles with start/end position marking")
    
    if results['valid_traversal_rate'] > 0.8:
        print("SUCCESS: Model respects terrain constraints")
    else:
        print("ISSUE: Model generates invalid paths through obstacles")
    
    if results['avg_total_path_tokens'] > results['avg_oracle_path_length'] * 2:
        print("ISSUE: Model outputs excessive PATH tokens (likely random scattering)")
    
    # Save results
    output_dir = Path("honest_evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump({k: float(v) if isinstance(v, np.floating) else v for k, v in results.items()}, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()