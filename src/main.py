"""
Main entry point for the battleground simulator application.
"""

import os
import sys
import torch
import numpy as np
import time

from src.simulation.simulator import BattleSimulator
from src.visualization.visualizer import BattlefieldVisualizer
from src.data.collector import BattleDataCollector
from src.models.strategy_recommender import StrategyRecommender, train_counter_strategy_model
from src.models.formation_recognizer import train_formation_recognizer
from src.models.environment import train_strategy_ai, evaluate_strategy
from src.utils.stats import display_battle_statistics, plot_win_rates, plot_formation_effectiveness
from src.utils.constants import FORMATION_RECOGNIZER_PATH, STRATEGY_MODEL_PATH, UNIT_TYPES


class BattlegroundSimulator:
    """Main class integrating all components of the battleground simulator."""
    
    def __init__(self):
        """Initialize the battleground simulator."""
        # Initialize components
        self.simulator = BattleSimulator()
        self.visualizer = BattlefieldVisualizer()
        self.data_collector = BattleDataCollector()
        
        # Load or initialize models
        self.load_models()
    
    def load_models(self):
        """Load AI models or initialize new ones if not available."""
        # Load strategy recommender
        self.strategy_recommender = StrategyRecommender(STRATEGY_MODEL_PATH)
        
        # Load PPO agent for reinforcement learning
        self.rl_agent = None
        try:
            # First check if stable_baselines3 is available
            print("Checking for stable-baselines3...")
            import sys
            
            try:
                import stable_baselines3
                print(f"Found stable_baselines3 version: {stable_baselines3.__version__}")
                from stable_baselines3 import PPO
                # Use 'MlpPolicy' as a string when creating PPO instances
                print("Successfully imported PPO")
                
                # Then check if the model file exists and has sufficient size
                if os.path.exists(STRATEGY_MODEL_PATH) and os.path.getsize(STRATEGY_MODEL_PATH) > 1000:
                    try:
                        print("Loading reinforcement learning model...")
                        self.rl_agent = PPO.load(STRATEGY_MODEL_PATH)
                        print("Reinforcement learning model loaded successfully")
                    except (ValueError, AssertionError) as e:
                        print(f"Warning: Could not load reinforcement learning model: {e}")
                        print("The model file may be corrupted or empty.")
                        print("You can train a new model using the 'Retrain Models' option.")
                else:
                    print("Reinforcement learning model not found or empty.")
                    print("You can train a model using the 'Retrain Models' option.")
            except ImportError as e:
                print(f"Error importing stable_baselines3: {e}")
                
                # Try to provide more helpful diagnostics
                try:
                    import pkg_resources
                    print("\nChecking installed packages:")
                    for pkg in pkg_resources.working_set:
                        if "stable" in pkg.key or "gym" in pkg.key:
                            print(f"  {pkg.key}: {pkg.version}")
                except Exception as ex:
                    print(f"Error checking packages: {ex}")
                
        except Exception as e:
            print(f"Note: Reinforcement learning features unavailable: {e}")
            print("To use reinforcement learning features, install: pip install stable-baselines3[extra]")
            print("This is optional - you can use all other simulator features without it.")
    
    def run_demo_battle(self, enemy_formation=None, use_rl=False):
        """
        Run a demonstration battle with visualization.
        
        Args:
            enemy_formation: Optional predefined enemy formation
            use_rl: Whether to use reinforcement learning model
            
        Returns:
            Tuple of (winner, enemy_health, home_health)
        """
        if enemy_formation is None:
            print("Generating random enemy formation...")
            enemy_formation = self.simulator.generate_random_formation("ENEMY")
        
        print("Analyzing enemy formation...")
        
        if use_rl:
            if self.rl_agent:
                try:
                    # Use reinforcement learning model
                    from gymnasium import spaces
                    
                    # Create flattened observation
                    observation = np.array(enemy_formation, dtype=np.float32)
                    
                    # Get action from RL model
                    action, _ = self.rl_agent.predict(observation)
                    
                    # Convert action to formation
                    home_formation = np.zeros_like(enemy_formation)
                    for unit_idx in range(action.shape[-1]):
                        for y in range(action.shape[0]):
                            for x in range(action.shape[1]):
                                if action[y, x, unit_idx] > 0.5:  # Threshold
                                    home_formation[y, x, unit_idx] = 1
                    
                    # Make the formation valid
                    home_formation = self.strategy_recommender._make_formation_valid(home_formation)
                    
                    print("Using reinforcement learning model for strategy")
                    
                    # Count unit types to show diversity information
                    unit_counts = {}
                    for unit_idx, unit_type in enumerate(UNIT_TYPES):
                        count = np.sum(home_formation[:, :, unit_idx] > 0)
                        if count > 0:
                            unit_counts[unit_type] = count
                    
                    # Display unit distribution
                    print("Unit distribution in RL formation:")
                    for unit_type, count in unit_counts.items():
                        print(f"  {unit_type}: {count}")
                        
                except Exception as e:
                    print(f"Error using reinforcement learning model: {e}")
                    print("Falling back to standard AI recommendation")
                    use_rl = False
            else:
                print("Reinforcement learning model not available.")
                print("To use RL features, install 'stable-baselines3' and train a model.")
                print("Falling back to standard AI recommendation")
                use_rl = False
        
        if not use_rl:
            # Use strategy recommender
            recommendations = self.strategy_recommender.recommend_formations(
                enemy_formation, num_recommendations=3
            )
            
            print("Recommended formations:")
            for i, rec in enumerate(recommendations):
                print(f"  {i+1}. Success probability: {rec['success_prob']:.2f}")
            
            # Use the best formation
            home_formation = recommendations[0]["formation"]
        
        # Simulate battle with history
        print("Simulating battle with history...")
        battle_history = self.simulator.simulate_battle_with_history(
            enemy_formation, home_formation
        )
        
        # Get final result
        final_state = battle_history[-1]
        enemy_health = final_state.calculate_total_health("ENEMY")
        home_health = final_state.calculate_total_health("HOME")
        
        if enemy_health > home_health:
            winner = "ENEMY"
        elif home_health > enemy_health:
            winner = "HOME"
        else:
            winner = "DRAW"
        
        # Visualize the battle (with option to skip to the end)
        print("Visualizing battle... (Press 'S' during replay to skip to the end)")
        self.visualizer.render_battle_replay(battle_history)
        
        print(f"Battle complete. Winner: {winner}")
        print(f"  Enemy remaining health: {enemy_health}")
        print(f"  Home remaining health: {home_health}")
        
        # Record data
        self.data_collector.record_battle(
            enemy_formation, home_formation, winner, enemy_health, home_health
        )
        
        return winner, enemy_health, home_health
    
    def run_multiple_test_battles(self, num_battles=10, use_rl=True):
        """
        Run multiple test battles to evaluate model performance.
        
        Args:
            num_battles: Number of battles to run
            use_rl: Whether to use reinforcement learning model
            
        Returns:
            Win rate and average unit diversity
        """
        print(f"Running {num_battles} test battles...")
        
        wins = 0
        total_unit_types_used = 0
        unit_type_counts = {unit_type: 0 for unit_type in UNIT_TYPES}
        
        start_time = time.time()
        
        for i in range(num_battles):
            print(f"Test battle {i+1}/{num_battles}")
            
            # Generate random enemy formation
            enemy_formation = self.simulator.generate_random_formation("ENEMY")
            
            # Use fast mode and skip replay for efficiency
            winner, _, _ = self.run_demo_battle(
                enemy_formation=enemy_formation, 
                use_rl=use_rl, 
                skip_replay=True,
                fast_mode=True
            )
            
            if winner == "HOME":
                wins += 1
                
            # Track unit types used if using RL
            if use_rl and self.rl_agent:
                observation = np.array(enemy_formation, dtype=np.float32)
                action, _ = self.rl_agent.predict(observation)
                
                # Count unit types in this formation
                unit_types_used = 0
                for unit_idx, unit_type in enumerate(UNIT_TYPES):
                    if np.any(action[:, :, unit_idx] > 0.5):
                        unit_types_used += 1
                        unit_type_counts[unit_type] += 1
                
                total_unit_types_used += unit_types_used
        
        win_rate = wins / num_battles
        avg_diversity = total_unit_types_used / num_battles if use_rl else 0
        
        total_time = time.time() - start_time
        
        print(f"\nTest results over {num_battles} battles:")
        print(f"Win rate: {win_rate:.2f} ({wins}/{num_battles})")
        
        if use_rl:
            print(f"Average unit types used per battle: {avg_diversity:.1f}")
            print("Total unit type usage distribution:")
            for unit_type, count in unit_type_counts.items():
                if count > 0:
                    print(f"  {unit_type}: {count} times ({count/num_battles:.1f} per battle)")
        
        print(f"Total time: {total_time:.1f} seconds ({total_time/num_battles:.1f} sec per battle)")
        
        return win_rate, avg_diversity
    
    def run_training_session(self, num_battles=100):
        """
        Run a training session with multiple battles.
        
        Args:
            num_battles: Number of battles to simulate
            
        Returns:
            Overall win rate
        """
        print(f"Starting training session with {num_battles} battles...")
        
        results = []
        for i in range(num_battles):
            print(f"Battle {i+1}/{num_battles}")
            
            # Generate random enemy formation
            enemy_formation = self.simulator.generate_random_formation("ENEMY")
            
            # Get AI recommendations
            recommendations = self.strategy_recommender.recommend_formations(
                enemy_formation, num_recommendations=3
            )
            
            # Simulate battles with each recommendation
            battle_results = []
            for rec in recommendations:
                winner, enemy_health, home_health = self.simulator.simulate_battle(
                    enemy_formation, rec["formation"]
                )
                
                battle_results.append({
                    "formation": rec["formation"],
                    "predicted_success": rec["success_prob"],
                    "actual_winner": winner,
                    "enemy_health": enemy_health,
                    "home_health": home_health
                })
                
                # Record data
                self.data_collector.record_battle(
                    enemy_formation, rec["formation"], winner, enemy_health, home_health
                )
            
            # Store result
            best_result = max(battle_results, key=lambda x: x["home_health"] - x["enemy_health"])
            results.append(best_result)
            
            # Display progress
            if (i+1) % 10 == 0:
                win_rate = sum(1 for r in results[-10:] if r["actual_winner"] == "HOME") / 10
                print(f"Last 10 battles win rate: {win_rate:.2f}")
        
        # Return overall results
        win_rate = sum(1 for r in results if r["actual_winner"] == "HOME") / len(results)
        print(f"Overall training session win rate: {win_rate:.2f}")
        
        return win_rate
    
    def retrain_models(self):
        """Retrain AI models with collected battle data."""
        print("Retraining AI models...")
        
        # Check if we have enough data
        battle_count = self.data_collector.get_battle_count()
        if battle_count < 20:
            print(f"Not enough battle data for training (only {battle_count} battles).")
            print("Run a training session first to collect data.")
            return
        
        print(f"Using {battle_count} battles for training.")
        
        # Retrain formation recognizer
        print("Training formation recognizer...")
        formation_recognizer = train_formation_recognizer(self.data_collector)
        if formation_recognizer:
            torch.save(formation_recognizer.state_dict(), FORMATION_RECOGNIZER_PATH)
            print("Formation recognizer trained and saved successfully")
        
        # Retrain strategy recommender
        print("Training counter-strategy model...")
        counter_strategy_model = train_counter_strategy_model(self.data_collector)
        if counter_strategy_model:
            print("Counter-strategy model trained and saved successfully")
        
        # Reinitialize strategy recommender with new models
        self.strategy_recommender = StrategyRecommender(STRATEGY_MODEL_PATH)
        
        # Optionally train reinforcement learning model
        train_rl = input("Would you like to train the reinforcement learning model?\nNote: This requires the 'stable-baselines3' package. (y/n): ")
        if train_rl.lower() == 'y':
            try:
                print("Training reinforcement learning model...")
                # Use 2000 iterations for proper exploration without adding bias
                self.rl_agent = train_strategy_ai(num_iterations=2000)
                if self.rl_agent:
                    print("Reinforcement learning model trained successfully")
                    print("Note: You can use the simulator without the reinforcement learning component.")
                else:
                    print("Note: You can use the simulator without the reinforcement learning component.")
            except Exception as e:
                print(f"Error training reinforcement learning model: {e}")
                print("You can continue using the simulator with the standard AI components.")
        else:
            print("Skipping reinforcement learning training.")
        
        print("Retraining complete")


def main():
    """Main entry point for the battleground simulator."""
    print("Battleground Simulator with AI Strategy")
    print("======================================")
    
    # Create simulator
    simulator = BattlegroundSimulator()
    
    # Main menu loop
    running = True
    while running:
        print("\nMain Menu:")
        print("1. Run Demo Battle")
        print("2. Run Training Session")
        print("3. Retrain Models")
        print("4. View Battle Statistics")
        print("5. Run Multiple Test Battles")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            rl_choice = input("Use reinforcement learning model? (y/n): ")
            use_rl = rl_choice.lower() == 'y'
            
            simulator.run_demo_battle(use_rl=use_rl)
        elif choice == "2":
            num_battles = int(input("Enter number of battles for training: "))
            simulator.run_training_session(num_battles)
        elif choice == "3":
            simulator.retrain_models()
        elif choice == "4":
            display_battle_statistics(simulator.data_collector)
            
            plot_choice = input("Show plots? (y/n): ")
            if plot_choice.lower() == 'y':
                plot_win_rates(simulator.data_collector)
                plot_formation_effectiveness(simulator.data_collector)
        elif choice == "5":
            # New option for running multiple test battles quickly
            num_battles = int(input("Enter number of test battles: "))
            rl_choice = input("Use reinforcement learning model? (y/n): ")
            use_rl = rl_choice.lower() == 'y'
            
            simulator.run_multiple_test_battles(num_battles=num_battles, use_rl=use_rl)
        elif choice == "6":
            running = False
        else:
            print("Invalid choice. Please try again.")
    
    print("Exiting Battleground Simulator. Goodbye!")
    
    # Close resources
    simulator.visualizer.close()
    simulator.data_collector.close()


if __name__ == "__main__":
    main() 