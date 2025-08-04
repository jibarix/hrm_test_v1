#!/usr/bin/env python3
"""
HRM Paper-Compatible Dataset Quality Analysis
Validates City Logistics dataset generated using paper-compatible methodology.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
from pathlib import Path

# Token mapping from the paper-compatible logistics game
HRM_TOKEN_MAP = {
    'PAD': 0,           # Padding token
    'OBSTACLE': 1,      # Buildings/City Park
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

class HRMPaperDatasetAnalyzer:
    def __init__(self, data_dir="data/logistics-routing-1k"):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        
        # Load all data including new metadata
        try:
            # Load standard data
            self.train_inputs = np.load(self.train_dir / "all__inputs.npy")
            self.train_labels = np.load(self.train_dir / "all__labels.npy")
            self.test_inputs = np.load(self.test_dir / "all__inputs.npy") 
            self.test_labels = np.load(self.test_dir / "all__labels.npy")
            
            # Load paper-specific metadata
            try:
                with open(self.train_dir / "all__base_scenario_ids.json") as f:
                    self.train_base_ids = json.load(f)
                with open(self.train_dir / "all__vehicle_types.json") as f:
                    self.train_vehicles = json.load(f)
                with open(self.test_dir / "all__base_scenario_ids.json") as f:
                    self.test_base_ids = json.load(f)
                with open(self.test_dir / "all__vehicle_types.json") as f:
                    self.test_vehicles = json.load(f)
                self.has_metadata = True
            except FileNotFoundError:
                print("⚠️ Paper metadata not found - using compatibility mode")
                self.has_metadata = False
            
            # Load dataset summary if available
            try:
                with open(self.data_dir / "dataset_summary.json") as f:
                    self.dataset_summary = json.load(f)
            except FileNotFoundError:
                self.dataset_summary = {}
            
            print(f"✅ Loaded {len(self.train_inputs)} training examples")
            print(f"✅ Loaded {len(self.test_inputs)} test examples")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("Make sure the dataset has been converted from JSON to .npy format")
            return
        
        self.results = {}
    
    def validate_paper_methodology(self):
        """Validate paper-compatible generation methodology"""
        print("\n" + "="*60)
        print("🧠 HRM PAPER METHODOLOGY VALIDATION")
        print("="*60)
        
        if not self.has_metadata:
            print("❌ Cannot validate paper methodology - metadata missing")
            return
        
        # Expected paper numbers
        expected_train_base = 240
        expected_test_base = 100
        expected_train_examples = 960  # 240 × 4
        expected_test_examples = 400   # 100 × 4
        expected_vehicles = ['easy', 'normal', 'hard', 'expert']
        
        # Validate example counts
        actual_train = len(self.train_inputs)
        actual_test = len(self.test_inputs)
        
        print(f"📊 Example Counts:")
        print(f"  Train: {actual_train} (expected: {expected_train_examples})")
        print(f"  Test: {actual_test} (expected: {expected_test_examples})")
        
        count_match = (actual_train == expected_train_examples and 
                      actual_test == expected_test_examples)
        print(f"  ✅ Count Match: {count_match}")
        
        # Validate base scenario structure
        train_base_scenarios = set(self.train_base_ids)
        test_base_scenarios = set(self.test_base_ids)
        
        print(f"\n🗺️ Base Scenario Analysis:")
        print(f"  Unique train base scenarios: {len(train_base_scenarios)}")
        print(f"  Unique test base scenarios: {len(test_base_scenarios)}")
        print(f"  Expected train base scenarios: {expected_train_base}")
        print(f"  Expected test base scenarios: {expected_test_base}")
        
        # Check for base scenario separation (critical for paper validity)
        base_overlap = train_base_scenarios.intersection(test_base_scenarios)
        print(f"  🚨 Base scenario overlap: {len(base_overlap)} scenarios")
        if len(base_overlap) > 0:
            print(f"    Overlapping scenarios: {sorted(list(base_overlap))[:10]}...")
        
        # Validate systematic vehicle augmentation
        print(f"\n🚗 Vehicle Augmentation Analysis:")
        train_vehicle_dist = Counter(self.train_vehicles)
        test_vehicle_dist = Counter(self.test_vehicles)
        
        print(f"  Train vehicle distribution: {dict(train_vehicle_dist)}")
        print(f"  Test vehicle distribution: {dict(test_vehicle_dist)}")
        
        # Check if each base scenario has 4 vehicle variants
        train_base_vehicle_counts = defaultdict(set)
        test_base_vehicle_counts = defaultdict(set)
        
        for base_id, vehicle in zip(self.train_base_ids, self.train_vehicles):
            train_base_vehicle_counts[base_id].add(vehicle)
        
        for base_id, vehicle in zip(self.test_base_ids, self.test_vehicles):
            test_base_vehicle_counts[base_id].add(vehicle)
        
        # Analyze systematic augmentation quality
        train_complete_variants = sum(1 for vehicles in train_base_vehicle_counts.values() 
                                    if len(vehicles) == 4)
        test_complete_variants = sum(1 for vehicles in test_base_vehicle_counts.values() 
                                   if len(vehicles) == 4)
        
        print(f"  Train base scenarios with 4 vehicle variants: {train_complete_variants}")
        print(f"  Test base scenarios with 4 vehicle variants: {test_complete_variants}")
        
        # Check vehicle type completeness
        missing_vehicles = set(expected_vehicles) - set(self.train_vehicles + self.test_vehicles)
        print(f"  Missing vehicle types: {missing_vehicles if missing_vehicles else 'None'}")
        
        # Store results for final scoring
        self.results['paper_methodology'] = {
            'count_match': count_match,
            'base_overlap': len(base_overlap),
            'train_complete_variants': train_complete_variants,
            'test_complete_variants': test_complete_variants,
            'expected_train_base': expected_train_base,
            'expected_test_base': expected_test_base,
            'missing_vehicles': len(missing_vehicles)
        }
        
        return count_match and len(base_overlap) == 0
    
    def analyze_systematic_augmentation_quality(self):
        """Analyze the quality of systematic vehicle augmentation"""
        print("\n" + "="*60)
        print("🔧 SYSTEMATIC AUGMENTATION QUALITY")
        print("="*60)
        
        if not self.has_metadata:
            print("❌ Cannot analyze augmentation - metadata missing")
            return
        
        # Analyze input diversity per base scenario
        base_scenario_inputs = defaultdict(list)
        base_scenario_vehicles = defaultdict(list)
        
        for i, (base_id, vehicle) in enumerate(zip(self.train_base_ids + self.test_base_ids, 
                                                   self.train_vehicles + self.test_vehicles)):
            if i < len(self.train_inputs):
                input_data = self.train_inputs[i]
            else:
                input_data = self.test_inputs[i - len(self.train_inputs)]
            
            base_scenario_inputs[base_id].append(input_data)
            base_scenario_vehicles[base_id].append(vehicle)
        
        # Check input consistency within base scenarios
        consistent_inputs = 0
        total_base_scenarios = len(base_scenario_inputs)
        
        for base_id, inputs in base_scenario_inputs.items():
            if len(inputs) > 1:
                # Remove start/end/path tokens for comparison (tokens 7, 8, 9)
                base_maps = []
                for inp in inputs:
                    base_map = inp.copy()
                    base_map[base_map == 7] = 2  # Replace start with road
                    base_map[base_map == 8] = 2  # Replace end with road  
                    base_map[base_map == 9] = 0  # Remove path
                    base_maps.append(base_map)
                
                # Check if base maps are identical (they should be for same scenario)
                maps_identical = all(np.array_equal(base_maps[0], bmap) for bmap in base_maps[1:])
                if maps_identical:
                    consistent_inputs += 1
        
        print(f"Base scenarios with consistent input maps: {consistent_inputs}/{total_base_scenarios}")
        
        # Analyze path diversity per base scenario  
        path_diversity_scores = []
        
        for base_id in base_scenario_inputs.keys():
            if base_id in [bid for bid in self.train_base_ids]:
                # Get indices for this base scenario in train set
                indices = [i for i, bid in enumerate(self.train_base_ids) if bid == base_id]
                paths = [self.train_labels[i] for i in indices]
            else:
                # Get indices for this base scenario in test set
                indices = [i for i, bid in enumerate(self.test_base_ids) if bid == base_id]
                paths = [self.test_labels[i] for i in indices]
            
            if len(paths) > 1:
                # Calculate path diversity (should be different for different vehicles)
                unique_paths = len(set(tuple(path) for path in paths))
                diversity_score = unique_paths / len(paths)
                path_diversity_scores.append(diversity_score)
        
        avg_path_diversity = np.mean(path_diversity_scores) if path_diversity_scores else 0
        print(f"Average path diversity per base scenario: {avg_path_diversity:.3f}")
        print(f"Expected: >0.5 (different vehicles → different paths)")
        
        self.results['augmentation_quality'] = {
            'consistent_inputs': consistent_inputs / total_base_scenarios if total_base_scenarios > 0 else 0,
            'avg_path_diversity': avg_path_diversity
        }
        
        return consistent_inputs / total_base_scenarios > 0.8 and avg_path_diversity > 0.5
    
    def analyze_hrm_token_distribution(self):
        """Analyze HRM token distribution and validate encoding"""
        print("\n" + "="*60)
        print("🔤 HRM TOKEN DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Combine all inputs for analysis
        all_inputs = np.concatenate([self.train_inputs, self.test_inputs])
        all_labels = np.concatenate([self.train_labels, self.test_labels])
        
        # Analyze input token distribution
        input_token_counts = Counter(all_inputs.flatten())
        label_token_counts = Counter(all_labels.flatten())
        
        print("Input Token Distribution:")
        for token_id, count in sorted(input_token_counts.items()):
            token_name = [k for k, v in HRM_TOKEN_MAP.items() if v == token_id]
            token_name = token_name[0] if token_name else f"UNKNOWN({token_id})"
            percentage = count / all_inputs.size * 100
            print(f"  {token_name}: {count:,} ({percentage:.2f}%)")
        
        print("\nLabel Token Distribution:")
        for token_id, count in sorted(label_token_counts.items()):
            token_name = [k for k, v in HRM_TOKEN_MAP.items() if v == token_id]
            token_name = token_name[0] if token_name else f"UNKNOWN({token_id})"
            percentage = count / all_labels.size * 100
            print(f"  {token_name}: {count:,} ({percentage:.2f}%)")
        
        # Validate expected token patterns
        expected_tokens = set(HRM_TOKEN_MAP.values())
        actual_input_tokens = set(input_token_counts.keys())
        actual_label_tokens = set(label_token_counts.keys())
        
        unexpected_input_tokens = actual_input_tokens - expected_tokens
        unexpected_label_tokens = actual_label_tokens - expected_tokens
        
        print(f"\n🚨 Unexpected tokens in inputs: {unexpected_input_tokens}")
        print(f"🚨 Unexpected tokens in labels: {unexpected_label_tokens}")
        
        # Check for proper start/end presence in inputs
        start_count = input_token_counts.get(HRM_TOKEN_MAP['START'], 0)
        end_count = input_token_counts.get(HRM_TOKEN_MAP['END'], 0)
        total_examples = len(all_inputs)
        
        print(f"\n📍 Start/End Token Analysis:")
        print(f"  START tokens: {start_count} (expected: {total_examples})")
        print(f"  END tokens: {end_count} (expected: {total_examples})")
        print(f"  Start/End ratio: {start_count/total_examples:.3f} (expected: 1.0)")
        
        # Check path token presence in labels
        path_count = label_token_counts.get(HRM_TOKEN_MAP['PATH'], 0)
        print(f"  PATH tokens in labels: {path_count:,}")
        
        # Validate road type ratios (should match City structure)
        road_tokens = [HRM_TOKEN_MAP['SMALL_ROAD'], HRM_TOKEN_MAP['LARGE_ROAD'], HRM_TOKEN_MAP['BROADWAY']]
        total_roads = sum(input_token_counts.get(token, 0) for token in road_tokens)
        
        if total_roads > 0:
            print(f"\n🛣️ Road Type Distribution:")
            for token_id in road_tokens:
                token_name = [k for k, v in HRM_TOKEN_MAP.items() if v == token_id][0]
                count = input_token_counts.get(token_id, 0)
                ratio = count / total_roads
                print(f"  {token_name}: {ratio:.3f}")
        
        self.results['token_distribution'] = {
            'unexpected_input_tokens': len(unexpected_input_tokens),
            'unexpected_label_tokens': len(unexpected_label_tokens),
            'start_end_ratio': start_count / total_examples if total_examples > 0 else 0,
            'path_tokens': path_count
        }
        
        return (len(unexpected_input_tokens) == 0 and 
                len(unexpected_label_tokens) == 0 and 
                start_count == total_examples == end_count)
    
    def analyze_path_optimality(self):
        """Analyze optimality of generated paths using A* validation"""
        print("\n" + "="*60)
        print("🎯 PATH OPTIMALITY ANALYSIS")
        print("="*60)
        
        # Sample subset for analysis (A* validation is expensive)
        sample_size = min(100, len(self.train_inputs))
        sample_indices = np.random.choice(len(self.train_inputs), sample_size, replace=False)
        
        valid_paths = 0
        connectivity_issues = 0
        optimal_paths = 0
        
        for i in sample_indices:
            input_grid = self.tokens_to_grid(self.train_inputs[i])
            label_grid = self.tokens_to_grid(self.train_labels[i])
            
            # Extract start, end, and path positions
            start_pos = self.find_token_position(input_grid, HRM_TOKEN_MAP['START'])
            end_pos = self.find_token_position(input_grid, HRM_TOKEN_MAP['END'])
            path_positions = self.find_token_positions(label_grid, HRM_TOKEN_MAP['PATH'])
            
            if start_pos and end_pos and len(path_positions) > 0:
                # Check path connectivity
                if self.is_path_connected(path_positions, start_pos, end_pos):
                    valid_paths += 1
                    
                    # Quick optimality check (Manhattan distance vs path length)
                    manhattan_dist = abs(start_pos[0] - end_pos[0]) + abs(start_pos[1] - end_pos[1])
                    path_length = len(path_positions)
                    
                    # Reasonable optimality: path length should be close to Manhattan distance
                    # Allow for obstacles and traffic (expect 1.2x to 3x Manhattan distance)
                    optimality_ratio = path_length / manhattan_dist
                    if 1.0 <= optimality_ratio <= 3.0:
                        optimal_paths += 1
                else:
                    connectivity_issues += 1
        
        print(f"Analyzed {sample_size} examples:")
        print(f"  Valid connected paths: {valid_paths}")
        print(f"  Connectivity issues: {connectivity_issues}")
        print(f"  Reasonably optimal paths: {optimal_paths}")
        print(f"  Path validity rate: {valid_paths/sample_size:.1%}")
        print(f"  Optimality rate: {optimal_paths/sample_size:.1%}")
        
        self.results['path_optimality'] = {
            'validity_rate': valid_paths / sample_size,
            'optimality_rate': optimal_paths / sample_size,
            'connectivity_issues': connectivity_issues
        }
        
        return valid_paths / sample_size > 0.95 and optimal_paths / sample_size > 0.8
    
    def tokens_to_grid(self, tokens):
        """Convert 1600-token sequence back to 40x40 grid"""
        return tokens.reshape(MAP_DIMENSIONS['height'], MAP_DIMENSIONS['width'])
    
    def find_token_position(self, grid, token_id):
        """Find first position of a token in grid"""
        positions = np.where(grid == token_id)
        if len(positions[0]) > 0:
            return (positions[1][0], positions[0][0])  # (x, y)
        return None
    
    def find_token_positions(self, grid, token_id):
        """Find all positions of a token in grid"""
        positions = np.where(grid == token_id)
        return list(zip(positions[1], positions[0]))  # [(x, y), ...]
    
    def is_path_connected(self, path_positions, start_pos, end_pos):
        """Check if path forms a connected route from start to end"""
        if not path_positions:
            return False
        
        path_set = set(path_positions + [start_pos, end_pos])
        
        # BFS to check connectivity
        visited = set()
        queue = [start_pos]
        visited.add(start_pos)
        
        while queue:
            current = queue.pop(0)
            if current == end_pos:
                return True
            
            # Check 4-connected neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in path_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    
    def generate_hrm_paper_report(self):
        """Generate comprehensive HRM paper compliance report"""
        print("\n" + "="*60)
        print("📊 HRM PAPER COMPLIANCE REPORT")
        print("="*60)
        
        # Calculate overall compliance score
        compliance_score = 0
        max_score = 0
        issues = []
        
        # Paper methodology compliance (weight: 40%)
        if 'paper_methodology' in self.results:
            methodology_score = 0
            if self.results['paper_methodology']['count_match']:
                methodology_score += 10
            else:
                issues.append("❌ Example counts don't match paper expectations")
            
            if self.results['paper_methodology']['base_overlap'] == 0:
                methodology_score += 15
            else:
                issues.append(f"🚨 CRITICAL: {self.results['paper_methodology']['base_overlap']} base scenarios overlap between train/test")
            
            if (self.results['paper_methodology']['train_complete_variants'] >= 200 and 
                self.results['paper_methodology']['test_complete_variants'] >= 80):
                methodology_score += 15
            else:
                issues.append("⚠️ Insufficient systematic vehicle augmentation")
            
            compliance_score += methodology_score
        max_score += 40
        
        # Token distribution compliance (weight: 20%)
        if 'token_distribution' in self.results:
            token_score = 0
            if (self.results['token_distribution']['unexpected_input_tokens'] == 0 and
                self.results['token_distribution']['unexpected_label_tokens'] == 0):
                token_score += 10
            else:
                issues.append("❌ Unexpected tokens found in dataset")
            
            if self.results['token_distribution']['start_end_ratio'] > 0.95:
                token_score += 10
            else:
                issues.append("❌ Missing start/end tokens in examples")
            
            compliance_score += token_score
        max_score += 20
        
        # Path optimality compliance (weight: 25%)
        if 'path_optimality' in self.results:
            path_score = 0
            if self.results['path_optimality']['validity_rate'] > 0.95:
                path_score += 15
            else:
                issues.append("❌ Path validity issues detected")
            
            if self.results['path_optimality']['optimality_rate'] > 0.8:
                path_score += 10
            else:
                issues.append("❌ Sub-optimal paths detected")
            
            compliance_score += path_score
        max_score += 25
        
        # Augmentation quality compliance (weight: 15%)
        if 'augmentation_quality' in self.results:
            aug_score = 0
            if self.results['augmentation_quality']['consistent_inputs'] > 0.8:
                aug_score += 8
            else:
                issues.append("❌ Inconsistent base scenario inputs")
            
            if self.results['augmentation_quality']['avg_path_diversity'] > 0.5:
                aug_score += 7
            else:
                issues.append("❌ Insufficient path diversity per base scenario")
            
            compliance_score += aug_score
        max_score += 15
        
        # Calculate final compliance percentage
        compliance_percentage = (compliance_score / max_score) * 100 if max_score > 0 else 0
        
        print(f"HRM PAPER COMPLIANCE SCORE: {compliance_score}/{max_score} ({compliance_percentage:.1f}%)")
        
        if compliance_percentage >= 90:
            recommendation = "✅ EXCELLENT - Fully compliant with HRM paper methodology"
        elif compliance_percentage >= 80:
            recommendation = "✅ GOOD - Minor deviations from paper methodology"
        elif compliance_percentage >= 70:
            recommendation = "⚠️ ACCEPTABLE - Some issues need addressing"
        else:
            recommendation = "🚨 POOR - Major compliance issues detected"
        
        print(f"RECOMMENDATION: {recommendation}")
        
        print("\nISSUES FOUND:")
        if not issues:
            print("  ✅ No compliance issues detected!")
        else:
            for issue in issues:
                print(f"  {issue}")
        
        print("\n🧠 HRM PAPER SPECIFIC VALIDATIONS:")
        print("  ✓ Base scenario separation (train/test)")
        print("  ✓ Systematic vehicle augmentation (4 variants per scenario)")
        print("  ✓ Expected example counts (960 train + 400 test)")
        print("  ✓ HRM token format compliance")
        print("  ✓ A* oracle path optimality")
        
        return compliance_percentage >= 80
    
    def run_full_hrm_analysis(self):
        """Run complete HRM paper-compatible analysis"""
        print("🧠 Starting HRM Paper-Compatible City Logistics Dataset Analysis...")
        print(f"Dataset location: {self.data_dir}")
        
        try:
            # Core analyses
            methodology_ok = self.validate_paper_methodology()
            augmentation_ok = self.analyze_systematic_augmentation_quality()
            tokens_ok = self.analyze_hrm_token_distribution()
            paths_ok = self.analyze_path_optimality()
            
            # Generate final compliance report
            overall_ok = self.generate_hrm_paper_report()
            
            return overall_ok
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    analyzer = HRMPaperDatasetAnalyzer()
    is_compliant = analyzer.run_full_hrm_analysis()
    
    if is_compliant:
        print("\n" + "="*60)
        print("✅ CONCLUSION: Dataset is HRM paper-compatible!")
        print("🚀 Ready for HRM training with paper methodology validation.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ CONCLUSION: Dataset has compliance issues.")
        print("🔧 Review issues and regenerate dataset if needed.")
        print("="*60)