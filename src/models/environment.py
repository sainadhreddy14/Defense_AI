"""
Reinforcement learning environment for strategy training.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import time
from tqdm import tqdm  # For progress bar

from ..simulation.simulator import BattleSimulator
from ..utils.constants import GRID_HEIGHT, UNIT_TYPES, UNIT_STATS


class BattlegroundEnv(gym.Env):
    """
    Custom Gym environment for the battleground simulator.
    Used for training the strategy AI with reinforcement learning.
    """
    
    def __init__(self):
        """Initialize the environment."""
        super(BattlegroundEnv, self).__init__()
        
        # Initialize battle simulator
        self.simulator = BattleSimulator()
        
        # Define action and observation spaces
        # Action space: Home formation (25 x 10 x 7)
        # We use continuous actions between 0 and 1 that will be processed by the environment
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(GRID_HEIGHT, 10, len(UNIT_TYPES)),
            dtype=np.float32
        )
        
        # Observation space: Enemy formation (25 x 10 x 7)
        self.observation_space = spaces.Box(
            low=0.0,
            high=float('inf'),  # Health values can be above 1
            shape=(GRID_HEIGHT, 10, len(UNIT_TYPES)),
            dtype=np.float32
        )
        
        # State variables
        self.enemy_formation = None
        self.current_step = 0
        self.max_steps = 100
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Generate a new random enemy formation
        self.enemy_formation = self.simulator.generate_random_formation("ENEMY")
        
        # Reset step counter
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Home formation (25 x 10 x 7) with values between 0 and 1
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to valid home formation
        home_formation = self._action_to_formation(action)
        
        # Run simulation
        winner, enemy_health, home_health = self.simulator.simulate_battle(
            self.enemy_formation, home_formation
        )
        
        # Calculate reward
        if winner == "HOME":
            # Winning is a big reward, scaled by how much health is left
            reward = 10.0 + (home_health / 5000.0)
        elif winner == "ENEMY":
            # Losing gives negative reward
            reward = -5.0
        else:  # DRAW
            # Draw gives small positive reward 
            reward = 0.1
        
        # Add a small reward based on the health difference
        health_diff = home_health - enemy_health
        reward += health_diff / 5000.0
        
        # Penalize for unnecessary units (efficiency)
        total_units = np.sum(home_formation > 0)
        if total_units > 0:
            efficiency = home_health / total_units
            reward += efficiency / 1000.0
        
        # Update step counter
        self.current_step += 1
        
        # Check if episode is done
        terminated = True  # Each episode is a single battle
        truncated = False  # No time limit, but we could add one
        
        # Return observation, reward, etc.
        return self._get_observation(), reward, terminated, truncated, {
            "winner": winner,
            "enemy_health": enemy_health,
            "home_health": home_health
        }
    
    def _get_observation(self):
        """
        Convert enemy formation to observation.
        
        Returns:
            Numpy array representing the observation
        """
        return np.array(self.enemy_formation, dtype=np.float32)
    
    def _action_to_formation(self, action):
        """
        Convert raw action (probabilities) to valid formation.
        
        Args:
            action: Numpy array with values between 0 and 1
            
        Returns:
            Valid formation array
        """
        # The strategy recommender has logic to convert a raw formation
        # to a valid one with proper health values and budget considerations
        # For simplicity, we can use this directly:
        
        # Create empty formation
        formation = np.zeros_like(action)
        
        # Track budget and units used
        remaining_budget = 1500
        unit_counts = {unit_type: 0 for unit_type in UNIT_TYPES}
        max_counts = {unit_type: UNIT_STATS[unit_type]["max"] for unit_type in UNIT_TYPES}
        
        # Flatten and sort by probability
        height, width, num_units = action.shape
        indices = []
        
        for unit_idx in range(num_units):
            unit_probs = action[:, :, unit_idx].flatten()
            sorted_indices = np.argsort(unit_probs)[::-1]  # Descending
            unit_type = UNIT_TYPES[unit_idx]
            
            # Add unit type info
            indices.extend([(idx // width, idx % width, unit_idx, unit_probs[idx]) 
                           for idx in sorted_indices])
        
        # Sort by overall probability
        indices.sort(key=lambda x: x[3], reverse=True)
        
        # Place units in order of probability until budget is exhausted
        for y, x, unit_idx, prob in indices:
            unit_type = UNIT_TYPES[unit_idx]
            
            # Skip if we've already placed a unit here
            if np.any(formation[y, x] > 0):
                continue
            
            # Skip if we've reached max units for this type
            if unit_counts[unit_type] >= max_counts[unit_type]:
                continue
            
            # Skip if we can't afford it
            unit_cost = UNIT_STATS[unit_type]["cost"]
            if unit_cost > remaining_budget:
                continue
            
            # Place unit
            formation[y, x, unit_idx] = UNIT_STATS[unit_type]["health"]
            remaining_budget -= unit_cost
            unit_counts[unit_type] += 1
            
            # Break if budget is exhausted
            if remaining_budget <= 0:
                break
        
        return formation
    
    def render(self, mode='human'):
        """Render the environment."""
        # No rendering is implemented for this environment
        pass
    
    def close(self):
        """Clean up resources."""
        pass


# Custom callback to show progress during training
class ProgressBarCallback:
    """
    Custom callback for tracking progress during training.
    """
    def __init__(self, total_timesteps):
        self.total_timesteps = total_timesteps
        self.current_step = 0
        self.pbar = None
        
    def __call__(self, locals_dict, globals_dict):
        if self.pbar is None:
            self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")
        
        # Update based on current step
        step_increase = locals_dict.get("self").num_timesteps - self.current_step
        self.current_step = locals_dict.get("self").num_timesteps
        self.pbar.update(step_increase)
        
        # Show some stats about every 100 steps
        if self.current_step % 100 == 0:
            ep_info_buffer = locals_dict.get("self").ep_info_buffer
            if len(ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in ep_info_buffer])
                mean_length = np.mean([ep_info["l"] for ep_info in ep_info_buffer])
                print(f"\nStep: {self.current_step}, Mean reward: {mean_reward:.2f}, Mean episode length: {mean_length:.2f}")
        
        # Check for training done
        if self.current_step >= self.total_timesteps:
            self.pbar.close()
        
        return True


# Helper function to train using PPO algorithm
def train_strategy_ai(num_iterations=10000):
    """
    Train the strategy AI using PPO.
    
    Args:
        num_iterations: Number of training steps
        
    Returns:
        Trained PPO agent or None if stable-baselines3 is not installed
    """
    print("Attempting to import stable-baselines3...")
    import sys
    print(f"Python path: {sys.executable}")
    print(f"Path environment: {sys.path}")
    
    try:
        import stable_baselines3
        print(f"Found stable_baselines3 version: {stable_baselines3.__version__}")
        from stable_baselines3 import PPO
        # In stable-baselines3 2.6.0, we use 'MlpPolicy' as a string instead of importing the class
        print("Successfully imported PPO")
    except ImportError as e:
        print(f"Error importing stable-baselines3: {e}")
        print("Please install it with: pip install stable-baselines3[extra]")
        print("This package is required for reinforcement learning but is optional for the simulator.")
        print("You can continue using other simulator features without it.")
        
        # Try to provide more helpful diagnostics
        try:
            import pkg_resources
            print("\nInstalled packages:")
            for pkg in pkg_resources.working_set:
                if "stable" in pkg.key or "gym" in pkg.key:
                    print(f"  {pkg.key}: {pkg.version}")
        except:
            pass
        
        return None
    
    print("Using stable-baselines3 for reinforcement learning...")
    
    try:
        # Try to import tqdm for progress bar
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            print("Note: Install 'tqdm' package for progress bars: pip install tqdm")
            has_tqdm = False
        
        # Create environment
        env = BattlegroundEnv()
        
        # Initialize PPO agent with 'MlpPolicy' as a string instead of a class reference
        agent = PPO(
            policy='MlpPolicy',  # Use string instead of class reference
            env=env,
            learning_rate=3e-4,
            # Use smaller batch sizes and steps for faster iteration
            n_steps=128,
            batch_size=32,
            n_epochs=5, 
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            # Increase entropy coefficient to encourage more exploration
            # This promotes trying different unit types without biasing the reward
            ent_coef=0.1,  # Higher value (default is 0.0) means more exploration
            verbose=0  # Set to 0 to use our custom progress tracking
        )
        
        # For better learning, make sure we have enough training steps
        # but allow the user to set a higher value if they want
        if num_iterations < 2000:
            print(f"Setting training iterations to 2000 for better exploration")
            num_iterations = 2000
            
        # Train
        print(f"Training agent for {num_iterations} steps...")
        
        start_time = time.time()
        
        # Create simple text-based progress indicator if tqdm is not available
        if has_tqdm:
            # Use our custom callback for progress tracking
            callback = ProgressBarCallback(num_iterations)
            agent.learn(total_timesteps=num_iterations, callback=callback)
        else:
            # Simple text-based progress reporting
            print("Training progress: ", end="")
            for i in range(10):
                agent.learn(total_timesteps=num_iterations//10)
                print(f"{(i+1)*10}%... ", end="", flush=True)
            print("Done!")
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        
        # Save model
        agent.save("strategy_ai_model")
        print("Reinforcement learning model saved as 'strategy_ai_model'")
        
        return agent
    except Exception as e:
        print(f"Error during reinforcement learning training: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_strategy(agent, num_battles=100):
    """
    Evaluate the strategy AI against random enemy formations.
    
    Args:
        agent: Trained PPO agent
        num_battles: Number of battles to simulate
        
    Returns:
        Win rate (between 0 and 1)
    """
    env = BattlegroundEnv()
    wins = 0
    
    for _ in range(num_battles):
        obs, _ = env.reset()
        action, _ = agent.predict(obs)
        _, reward, _, _, info = env.step(action)
        
        if info["winner"] == "HOME":
            wins += 1
    
    return wins / num_battles 