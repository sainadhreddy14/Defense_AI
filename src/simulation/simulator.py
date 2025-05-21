"""
Battle simulator that manages the overall simulation process.
"""

import numpy as np
import copy
from src.simulation.battlefield import Battlefield
from src.utils.constants import UNIT_STATS, UNIT_TO_IDX, UNIT_TYPES, MAX_BUDGET, SIDES


class BattleSimulator:
    """
    Manages the simulation of battles between opposing formations.
    """
    
    def __init__(self, max_turns=100):
        """
        Initialize the battle simulator.
        
        Args:
            max_turns: Maximum number of turns for a battle before forced completion
        """
        self.max_turns = max_turns
    
    def simulate_battle(self, enemy_formation, home_formation):
        """
        Simulate a battle between enemy and home formations.
        
        Args:
            enemy_formation: 3D numpy array representing enemy formation
            home_formation: 3D numpy array representing home formation
            
        Returns:
            Tuple of (winner, enemy_health, home_health)
        """
        # Initialize battlefield with both formations
        battlefield = Battlefield()
        battlefield.apply_formation(enemy_formation, "ENEMY")
        battlefield.apply_formation(home_formation, "HOME")
        
        # Run simulation until completion
        turn = 0
        while not battlefield.is_battle_complete() and turn < self.max_turns:
            # Move units
            units_moved = battlefield.move_units()
            
            # Resolve combat
            combat_count = battlefield.resolve_combat()
            
            # If no movement and no combat, break early
            if units_moved == 0 and combat_count == 0:
                break
            
            turn += 1
        
        # Calculate results
        enemy_health = battlefield.calculate_total_health("ENEMY")
        home_health = battlefield.calculate_total_health("HOME")
        
        # Determine winner
        if enemy_health > home_health:
            winner = "ENEMY"
        elif home_health > enemy_health:
            winner = "HOME"
        else:
            winner = "DRAW"
        
        return winner, enemy_health, home_health
    
    def simulate_battle_with_history(self, enemy_formation, home_formation):
        """
        Simulate a battle and return the full history of battlefield states.
        
        Args:
            enemy_formation: 3D numpy array representing enemy formation
            home_formation: 3D numpy array representing home formation
            
        Returns:
            List of battlefield states (copies) at each turn
        """
        # Initialize battlefield with both formations
        battlefield = Battlefield()
        battlefield.apply_formation(enemy_formation, "ENEMY")
        battlefield.apply_formation(home_formation, "HOME")
        
        # Store history
        history = [battlefield.clone()]
        
        # Run simulation
        turn = 0
        while not battlefield.is_battle_complete() and turn < self.max_turns:
            # Move units
            units_moved = battlefield.move_units()
            
            # Store state after movement
            history.append(battlefield.clone())
            
            # Resolve combat
            combat_count = battlefield.resolve_combat()
            
            # Store state after combat
            history.append(battlefield.clone())
            
            # If no movement and no combat, break early
            if units_moved == 0 and combat_count == 0:
                break
            
            turn += 1
        
        return history
    
    def generate_random_formation(self, side="ENEMY", budget=MAX_BUDGET):
        """
        Generate a random but valid formation within budget constraints.
        
        Args:
            side: 'ENEMY' or 'HOME'
            budget: Maximum budget for the formation
            
        Returns:
            3D numpy array representing the formation
        """
        if side not in SIDES:
            raise ValueError(f"Invalid side: {side}")
        
        # Determine base width based on side
        base_width = 10  # Same for both sides
        
        # Create empty formation
        formation = np.zeros((25, base_width, len(UNIT_TYPES)))
        
        # Track remaining budget and unit counts
        remaining_budget = budget
        unit_counts = {unit_type: 0 for unit_type in UNIT_TYPES}
        
        # Generate random units until budget is exhausted
        attempts = 0
        while remaining_budget > 0 and attempts < 1000:
            # Pick a random unit type
            unit_type = np.random.choice(UNIT_TYPES)
            
            # Check if we've reached the maximum for this unit type
            if unit_counts[unit_type] >= UNIT_STATS[unit_type]["max"]:
                attempts += 1
                continue
            
            # Check if we can afford it
            unit_cost = UNIT_STATS[unit_type]["cost"]
            if unit_cost > remaining_budget:
                attempts += 1
                continue
            
            # Find a random empty position
            while True:
                x = np.random.randint(0, base_width)
                y = np.random.randint(0, 25)
                
                # Check if position is empty
                if np.all(formation[y, x] == 0):
                    # Place unit
                    unit_idx = UNIT_TO_IDX[unit_type]
                    formation[y, x, unit_idx] = UNIT_STATS[unit_type]["health"]
                    
                    # Update tracking
                    remaining_budget -= unit_cost
                    unit_counts[unit_type] += 1
                    break
            
            attempts = 0  # Reset attempts after successful placement
        
        return formation
    
    def validate_formation(self, formation, budget=MAX_BUDGET):
        """
        Check if a formation is valid according to rules and budget.
        
        Args:
            formation: 3D numpy array representing the formation
            budget: Maximum budget for the formation
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check dimensions
        if formation.shape[0] != 25 or formation.shape[1] != 10 or formation.shape[2] != len(UNIT_TYPES):
            return False, "Invalid formation dimensions"
        
        # Calculate cost and check unit limits
        total_cost = 0
        unit_counts = {unit_type: 0 for unit_type in UNIT_TYPES}
        
        for y in range(25):
            for x in range(10):
                for unit_idx, unit_type in enumerate(UNIT_TYPES):
                    # Check if unit exists at this position
                    if formation[y, x, unit_idx] > 0:
                        # Check for multiple units in same cell
                        if np.sum(formation[y, x] > 0) > 1:
                            return False, "Multiple units in same cell"
                        
                        # Update count
                        unit_counts[unit_type] += 1
                        
                        # Check unit limit
                        if unit_counts[unit_type] > UNIT_STATS[unit_type]["max"]:
                            return False, f"Exceeded maximum {unit_type} count"
                        
                        # Add to cost
                        total_cost += UNIT_STATS[unit_type]["cost"]
        
        # Check budget
        if total_cost > budget:
            return False, f"Formation cost ({total_cost}) exceeds budget ({budget})"
        
        return True, "Valid formation" 