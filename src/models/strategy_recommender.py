"""
AI Strategy recommendation system that suggests counter-formations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader

from .formation_recognizer import FormationRecognizer
from ..utils.constants import (
    UNIT_TYPES, UNIT_STATS, GRID_HEIGHT, MAX_BUDGET,
    STRATEGY_MODEL_PATH, FORMATION_RECOGNIZER_PATH, LEARNING_RATE
)
from ..strategies.patterns import (
    create_line_formation, create_wedge_formation,
    create_echelon_formation, create_refused_flank_formation
)


class CounterStrategyDataset(Dataset):
    """Dataset for training the counter-strategy model."""
    
    def __init__(self, enemy_formations, home_formations):
        """
        Initialize dataset with pairs of enemy and home formations.
        
        Args:
            enemy_formations: List of enemy formations (numpy arrays)
            home_formations: List of corresponding home formations
        """
        self.enemy_formations = enemy_formations
        self.home_formations = home_formations
    
    def __len__(self):
        """Return the number of formation pairs."""
        return len(self.enemy_formations)
    
    def __getitem__(self, idx):
        """Get a pair of formations by index."""
        enemy = self.enemy_formations[idx]
        home = self.home_formations[idx]
        
        # Convert to PyTorch tensors
        enemy_tensor = torch.tensor(enemy, dtype=torch.float32)
        home_tensor = torch.tensor(home, dtype=torch.float32)
        
        return enemy_tensor, home_tensor


class CounterStrategyGenerator(nn.Module):
    """
    Neural network model that generates counter-formations.
    Takes an enemy formation embedding and generates a counter-formation.
    """
    
    def __init__(self, embedding_size=64):
        """
        Initialize the counter-strategy generator model.
        
        Args:
            embedding_size: Size of the formation embedding vector
        """
        super(CounterStrategyGenerator, self).__init__()
        
        # Input: formation embedding vector
        # Output: counter-formation (25 x 10 x num_unit_types)
        
        # Fully connected layers
        self.fc1 = nn.Linear(embedding_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        
        # Reshape to 3D
        # Calculate dimensions for reshaping
        # Want to get to (height, width, channels) = (25, 10, len(UNIT_TYPES))
        # via upsampling, so we start with smaller dimensions
        self.height_factor = 6  # Will upsample to 24 (close to 25)
        self.width_factor = 3   # Will upsample to 9 (close to 10)
        self.reshape_size = self.height_factor * self.width_factor * 32
        
        self.fc4 = nn.Linear(1024, self.reshape_size)
        
        # Transposed convolutions for upsampling
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # 6x3 -> 12x6
        self.deconv2 = nn.ConvTranspose2d(16, len(UNIT_TYPES), kernel_size=2, stride=2)  # 12x6 -> 24x12
        
        # Final adjustment to match exact shape if needed
        self.final_conv = nn.Conv2d(len(UNIT_TYPES), len(UNIT_TYPES), kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass to generate a counter-formation.
        
        Args:
            x: Formation embedding tensor
            
        Returns:
            Generated counter-formation
        """
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        # Reshape for transposed convolutions
        batch_size = x.size(0)
        x = x.view(batch_size, 32, self.height_factor, self.width_factor)
        
        # Upsample
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))  # Sigmoid for 0-1 values
        
        # Apply final adjustment
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        
        # We need to crop/pad to get the exact dimensions
        # Crop or pad height to match GRID_HEIGHT
        height_diff = GRID_HEIGHT - x.size(2)
        if height_diff > 0:
            # Pad
            x = F.pad(x, (0, 0, 0, height_diff))
        elif height_diff < 0:
            # Crop
            x = x[:, :, :GRID_HEIGHT, :]
        
        # Crop or pad width to match 10
        width_diff = 10 - x.size(3)
        if width_diff > 0:
            # Pad
            x = F.pad(x, (0, width_diff))
        elif width_diff < 0:
            # Crop
            x = x[:, :, :, :10]
        
        # Rearrange from (batch, channels, height, width) to (batch, height, width, channels)
        x = x.permute(0, 2, 3, 1)
        
        return x


class StrategyRecommender:
    """
    System that recommends counter-formations for enemy formations.
    Uses both neural models and rule-based strategies.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the strategy recommender.
        
        Args:
            model_path: Path to the pre-trained model
        """
        # Load formation recognizer
        self.formation_recognizer = FormationRecognizer()
        try:
            if os.path.exists(FORMATION_RECOGNIZER_PATH):
                self.formation_recognizer.load_state_dict(
                    torch.load(FORMATION_RECOGNIZER_PATH)
                )
                print("Loaded formation recognizer model")
        except Exception as e:
            print(f"Could not load formation recognizer: {e}")
        
        # Initialize counter-strategy generator
        self.counter_strategy_generator = CounterStrategyGenerator()
        try:
            if model_path and os.path.exists(model_path):
                self.counter_strategy_generator.load_state_dict(
                    torch.load(model_path)
                )
                print("Loaded counter-strategy generator model")
        except Exception as e:
            print(f"Could not load counter-strategy generator: {e}")
        
        # Set models to evaluation mode
        self.formation_recognizer.eval()
        self.counter_strategy_generator.eval()
    
    def recommend_formations(self, enemy_formation, num_recommendations=5):
        """
        Generate recommended counter-formations for an enemy formation.
        
        Args:
            enemy_formation: 3D numpy array representing enemy formation
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of dicts with formations and success probabilities
        """
        recommendations = []
        
        # Get embedding of enemy formation
        enemy_embedding = self.formation_recognizer.get_embedding(enemy_formation)
        enemy_embedding_tensor = torch.tensor(enemy_embedding, dtype=torch.float32).unsqueeze(0)
        
        # Generate a few recommendations from the neural model
        if random.random() < 0.7:  # 70% chance to use neural model
            with torch.no_grad():
                for _ in range(min(3, num_recommendations)):
                    # Add some noise to embedding for variety
                    noise = torch.randn_like(enemy_embedding_tensor) * 0.05
                    noisy_embedding = enemy_embedding_tensor + noise
                    
                    # Generate counter-formation
                    generated = self.counter_strategy_generator(noisy_embedding)
                    formation_array = generated.squeeze(0).numpy()
                    
                    # Process the formation to make it valid
                    valid_formation = self._make_formation_valid(formation_array)
                    
                    # Estimate success probability
                    success_prob = self._estimate_success_probability(enemy_formation, valid_formation)
                    
                    recommendations.append({
                        "formation": valid_formation,
                        "success_prob": success_prob,
                        "source": "neural"
                    })
        
        # Add some rule-based formations
        rule_based_formations = self._generate_rule_based_formations(
            enemy_formation, 
            num_recommendations - len(recommendations)
        )
        
        recommendations.extend(rule_based_formations)
        
        # Sort by success probability
        recommendations.sort(key=lambda x: x["success_prob"], reverse=True)
        
        return recommendations[:num_recommendations]
    
    def _make_formation_valid(self, formation):
        """
        Process a raw generated formation to ensure it's valid.
        
        Args:
            formation: 3D numpy array representing a formation
            
        Returns:
            Valid formation
        """
        # Create empty formation
        valid_formation = np.zeros_like(formation)
        
        # Track budget and unit counts
        remaining_budget = MAX_BUDGET
        unit_counts = {unit_type: 0 for unit_type in UNIT_TYPES}
        
        # Find cells with highest probability for each unit type
        # Create flattened view with indices
        height, width, num_units = formation.shape
        flat_indices = []
        
        for unit_idx in range(num_units):
            # Get probability for this unit type across all cells
            unit_probs = formation[:, :, unit_idx].flatten()
            
            # Sort indices by probability
            sorted_indices = np.argsort(unit_probs)[::-1]  # Descending
            
            # Convert flat indices to (y, x) coordinates
            coords = [(idx // width, idx % width) for idx in sorted_indices]
            
            # Add unit type
            flat_indices.extend([(y, x, unit_idx) for y, x in coords])
        
        # Sort by overall probability
        flat_indices.sort(key=lambda idx: formation[idx[0], idx[1], idx[2]], reverse=True)
        
        # Place units in order of probability until budget is exhausted
        for y, x, unit_idx in flat_indices:
            unit_type = UNIT_TYPES[unit_idx]
            
            # Skip if we've already placed a unit here
            if np.any(valid_formation[y, x] > 0):
                continue
            
            # Skip if we've reached max units for this type
            if unit_counts[unit_type] >= UNIT_STATS[unit_type]["max"]:
                continue
            
            # Skip if we can't afford it
            unit_cost = UNIT_STATS[unit_type]["cost"]
            if unit_cost > remaining_budget:
                continue
            
            # Place unit
            valid_formation[y, x, unit_idx] = UNIT_STATS[unit_type]["health"]
            remaining_budget -= unit_cost
            unit_counts[unit_type] += 1
            
            # Break if budget is exhausted
            if remaining_budget <= 0:
                break
        
        return valid_formation
    
    def _generate_rule_based_formations(self, enemy_formation, count):
        """
        Generate rule-based formations as counter-strategies.
        
        Args:
            enemy_formation: Enemy formation to counter
            count: Number of formations to generate
            
        Returns:
            List of formation recommendations
        """
        recommendations = []
        
        # Analyze enemy formation to determine response
        formation_type = self._analyze_formation(enemy_formation)
        
        # Create unit mix based on enemy formation
        unit_mix = self._create_counter_unit_mix(enemy_formation)
        
        # Generate different formation patterns
        if formation_type == "TOP_HEAVY":
            # Counter with bottom-heavy formation
            bottom_heavy = create_refused_flank_formation(unit_mix, strong_side="bottom")
            recommendations.append({
                "formation": bottom_heavy,
                "success_prob": 0.8,
                "source": "rule_based"
            })
        
        elif formation_type == "BOTTOM_HEAVY":
            # Counter with top-heavy formation
            top_heavy = create_refused_flank_formation(unit_mix, strong_side="top")
            recommendations.append({
                "formation": top_heavy,
                "success_prob": 0.8,
                "source": "rule_based"
            })
        
        elif formation_type == "BALANCED":
            # Use wedge formation to break through center
            wedge = create_wedge_formation(unit_mix)
            recommendations.append({
                "formation": wedge,
                "success_prob": 0.75,
                "source": "rule_based"
            })
        
        elif formation_type == "LEFT_CONCENTRATED":
            # Counter with right echelon
            right_echelon = create_echelon_formation(unit_mix, right_sided=True)
            recommendations.append({
                "formation": right_echelon,
                "success_prob": 0.85,
                "source": "rule_based"
            })
        
        elif formation_type == "RIGHT_CONCENTRATED":
            # Counter with left echelon
            left_echelon = create_echelon_formation(unit_mix, right_sided=False)
            recommendations.append({
                "formation": left_echelon,
                "success_prob": 0.85,
                "source": "rule_based"
            })
        
        # Always add a line formation as a fallback
        line_formation = create_line_formation(unit_mix)
        recommendations.append({
            "formation": line_formation,
            "success_prob": 0.7,
            "source": "rule_based"
        })
        
        # Randomize success probabilities slightly to add variety
        for rec in recommendations:
            rec["success_prob"] += random.uniform(-0.1, 0.1)
            rec["success_prob"] = max(0.5, min(0.95, rec["success_prob"]))
        
        # Sort by success probability
        recommendations.sort(key=lambda x: x["success_prob"], reverse=True)
        
        return recommendations[:count]
    
    def _analyze_formation(self, formation):
        """
        Analyze an enemy formation to determine its type.
        
        Args:
            formation: 3D numpy array representing the formation
            
        Returns:
            String representing formation type
        """
        height, width, _ = formation.shape
        
        # Count units in different regions
        top_count = np.sum(formation[:height//3, :, :] > 0)
        middle_count = np.sum(formation[height//3:2*height//3, :, :] > 0)
        bottom_count = np.sum(formation[2*height//3:, :, :] > 0)
        
        left_count = np.sum(formation[:, :width//2, :] > 0)
        right_count = np.sum(formation[:, width//2:, :] > 0)
        
        # Determine formation type based on distribution
        if top_count > middle_count + bottom_count:
            return "TOP_HEAVY"
        elif bottom_count > middle_count + top_count:
            return "BOTTOM_HEAVY"
        elif abs(top_count - bottom_count) < height//4 and (top_count + bottom_count) > 1.5 * middle_count:
            return "FLANKING"
        elif left_count > 2 * right_count:
            return "LEFT_CONCENTRATED"
        elif right_count > 2 * left_count:
            return "RIGHT_CONCENTRATED"
        elif abs(top_count - middle_count - bottom_count) < height//5:
            return "BALANCED"
        else:
            return "MIXED"
    
    def _create_counter_unit_mix(self, enemy_formation):
        """
        Create a unit mix that counters the enemy formation.
        
        Args:
            enemy_formation: 3D numpy array representing enemy formation
            
        Returns:
            Dictionary mapping unit types to counts
        """
        # Count enemy units by type
        enemy_units = {}
        for unit_idx, unit_type in enumerate(UNIT_TYPES):
            count = np.sum(enemy_formation[:, :, unit_idx] > 0)
            enemy_units[unit_type] = count
        
        # Create counter mix based on enemy composition
        counter_mix = {unit_type: 0 for unit_type in UNIT_TYPES}
        remaining_budget = MAX_BUDGET
        
        # Counter logic
        if enemy_units["TANK"] > 5:
            # Lots of tanks, use artillery and fighter jets
            counter_mix["ARTILLERY"] = min(enemy_units["TANK"], UNIT_STATS["ARTILLERY"]["max"])
            counter_mix["FIGHTER_JET"] = min(5, UNIT_STATS["FIGHTER_JET"]["max"])
        else:
            # Fewer tanks, use our own tanks and landmines
            counter_mix["TANK"] = min(enemy_units["TANK"] + 2, UNIT_STATS["TANK"]["max"])
            counter_mix["LANDMINE"] = min(5, UNIT_STATS["LANDMINE"]["max"])
        
        if enemy_units["FIGHTER_JET"] > 3:
            # Lots of air units, use guard towers
            counter_mix["GUARD_TOWER"] = min(enemy_units["FIGHTER_JET"], UNIT_STATS["GUARD_TOWER"]["max"])
        
        if enemy_units["SOLDIER"] + enemy_units["SHIELDED_SOLDIER"] > 20:
            # Lots of infantry, use artillery
            counter_mix["ARTILLERY"] = min(10, UNIT_STATS["ARTILLERY"]["max"])
        else:
            # Fewer infantry, use our own infantry
            counter_mix["SHIELDED_SOLDIER"] = min(
                enemy_units["SOLDIER"] + enemy_units["SHIELDED_SOLDIER"],
                UNIT_STATS["SHIELDED_SOLDIER"]["max"]
            )
        
        # Calculate cost so far
        current_cost = sum(counter_mix[unit] * UNIT_STATS[unit]["cost"] for unit in UNIT_TYPES)
        remaining_budget = MAX_BUDGET - current_cost
        
        # Fill remaining budget with soldiers and tanks
        while remaining_budget >= UNIT_STATS["SOLDIER"]["cost"]:
            if (remaining_budget >= UNIT_STATS["TANK"]["cost"] and
                counter_mix["TANK"] < UNIT_STATS["TANK"]["max"] and
                random.random() < 0.3):
                # Add a tank with 30% probability if we can afford it
                counter_mix["TANK"] += 1
                remaining_budget -= UNIT_STATS["TANK"]["cost"]
            elif counter_mix["SOLDIER"] < UNIT_STATS["SOLDIER"]["max"]:
                # Add a soldier
                counter_mix["SOLDIER"] += 1
                remaining_budget -= UNIT_STATS["SOLDIER"]["cost"]
            else:
                # Can't add more soldiers
                break
        
        return counter_mix
    
    def _estimate_success_probability(self, enemy_formation, home_formation):
        """
        Estimate the probability of success for a counter-formation.
        
        Args:
            enemy_formation: Enemy formation
            home_formation: Counter-formation
            
        Returns:
            Estimated probability of success (0.0 to 1.0)
        """
        # Count units and calculate total stats
        enemy_stats = self._calculate_formation_stats(enemy_formation)
        home_stats = self._calculate_formation_stats(home_formation)
        
        # Simple formula based on relative attack and health
        attack_ratio = home_stats["total_attack"] / max(1, enemy_stats["total_attack"])
        health_ratio = home_stats["total_health"] / max(1, enemy_stats["total_health"])
        
        # Count specific counters (e.g., artillery vs tanks)
        counter_bonus = 0
        
        if enemy_stats["unit_counts"]["TANK"] > 0 and home_stats["unit_counts"]["ARTILLERY"] > 0:
            counter_bonus += 0.05 * min(enemy_stats["unit_counts"]["TANK"], home_stats["unit_counts"]["ARTILLERY"])
        
        if enemy_stats["unit_counts"]["FIGHTER_JET"] > 0 and home_stats["unit_counts"]["GUARD_TOWER"] > 0:
            counter_bonus += 0.05 * min(enemy_stats["unit_counts"]["FIGHTER_JET"], home_stats["unit_counts"]["GUARD_TOWER"])
        
        # Calculate base probability
        base_prob = 0.5 * (attack_ratio + health_ratio)
        
        # Apply bonus and ensure result is between 0.1 and 0.9
        prob = base_prob + counter_bonus
        prob = max(0.1, min(0.9, prob))
        
        return prob
    
    def _calculate_formation_stats(self, formation):
        """
        Calculate statistics for a formation.
        
        Args:
            formation: 3D numpy array representing formation
            
        Returns:
            Dictionary with formation statistics
        """
        stats = {
            "total_attack": 0,
            "total_health": 0,
            "unit_counts": {unit_type: 0 for unit_type in UNIT_TYPES}
        }
        
        for unit_idx, unit_type in enumerate(UNIT_TYPES):
            # Count units
            unit_mask = formation[:, :, unit_idx] > 0
            unit_count = np.sum(unit_mask)
            stats["unit_counts"][unit_type] = unit_count
            
            # Add to totals
            stats["total_attack"] += unit_count * UNIT_STATS[unit_type]["attack"]
            stats["total_health"] += np.sum(formation[:, :, unit_idx])
        
        return stats


def train_counter_strategy_model(data_collector, epochs=50):
    """
    Train a model to predict effective counter-strategies.
    
    Args:
        data_collector: BattleDataCollector instance
        epochs: Number of training epochs
        
    Returns:
        Trained CounterStrategyGenerator model
    """
    # Get training data
    battles = data_collector.get_training_data(limit=10000)
    
    # Filter for winning home formations
    winning_battles = [b for b in battles if b["winner"] == "HOME"]
    
    if len(winning_battles) < 10:
        print("Not enough winning battles for training. Defaulting to all battles.")
        winning_battles = battles
    
    # Prepare training pairs: (enemy_formation, winning_home_formation)
    X = [b["enemy_formation"] for b in winning_battles]
    y = [b["home_formation"] for b in winning_battles]
    
    # Load formation recognizer for embeddings
    formation_recognizer = FormationRecognizer()
    try:
        if os.path.exists(FORMATION_RECOGNIZER_PATH):
            formation_recognizer.load_state_dict(torch.load(FORMATION_RECOGNIZER_PATH))
    except Exception as e:
        print(f"Could not load formation recognizer: {e}")
    
    formation_recognizer.eval()
    
    # Extract embeddings
    X_embeddings = [formation_recognizer.get_embedding(x) for x in X]
    
    # Create custom dataset and dataloader
    dataset = list(zip(X_embeddings, y))
    dataloader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=True, 
        collate_fn=lambda batch: (
            torch.tensor([item[0] for item in batch], dtype=torch.float32),
            torch.tensor([item[1] for item in batch], dtype=torch.float32)
        )
    )
    
    # Initialize model
    model = CounterStrategyGenerator()
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for enemy_embeddings, home_formations in dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            predicted_formations = model(enemy_embeddings)
            
            # Calculate loss
            loss = F.mse_loss(predicted_formations, home_formations)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Save model
    torch.save(model.state_dict(), STRATEGY_MODEL_PATH)
    
    # Set model to evaluation mode
    model.eval()
    
    return model


if __name__ == "__main__":
    # Simple test of model architecture
    model = CounterStrategyGenerator()
    
    # Create a random embedding
    embedding = np.random.rand(64)
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    
    # Forward pass
    formation = model(embedding_tensor)
    
    print(f"Input shape: {embedding_tensor.shape}")
    print(f"Output shape: {formation.shape}")
    
    # Test recommender
    recommender = StrategyRecommender()
    
    # Create a random enemy formation
    enemy_formation = np.zeros((25, 10, len(UNIT_TYPES)))
    for _ in range(10):
        x = random.randint(0, 9)
        y = random.randint(0, 24)
        unit_idx = random.randint(0, len(UNIT_TYPES) - 1)
        enemy_formation[y, x, unit_idx] = UNIT_STATS[UNIT_TYPES[unit_idx]]["health"]
    
    # Get recommendations
    recommendations = recommender.recommend_formations(enemy_formation, num_recommendations=3)
    
    print(f"Generated {len(recommendations)} recommendations")
    for i, rec in enumerate(recommendations):
        print(f"Recommendation {i+1}: Success probability = {rec['success_prob']:.2f}, Source = {rec['source']}") 