"""
Core battlefield management and simulation logic.
"""

import numpy as np
import copy
from src.utils.constants import (
    GRID_WIDTH, GRID_HEIGHT, ENEMY_BASE_WIDTH, HOME_BASE_WIDTH,
    UNIT_STATS, UNIT_TO_IDX, UNIT_TYPES, SIDES
)


class Battlefield:
    """
    Represents and manages the battlefield state.
    
    The battlefield is represented as a 3D grid:
    - 1st dimension: height (25 squares)
    - 2nd dimension: width (40 squares)
    - 3rd dimension: unit type channels (7 types)
    
    Each cell can contain a unit, represented by a value indicating its health.
    """
    
    def __init__(self):
        """Initialize an empty battlefield."""
        # Main grid - shape (25, 40, 7)
        # Each cell is the health of a unit (or 0 if no unit)
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(UNIT_TYPES)))
        
        # Ownership grid - shape (25, 40)
        # 0 = no owner, 1 = enemy, 2 = home
        self.ownership = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int8)
        
        # Setup base regions
        self.setup_base_regions()
    
    def setup_base_regions(self):
        """Define the base regions on the ownership grid."""
        # Enemy base (left side)
        self.ownership[:, :ENEMY_BASE_WIDTH] = 1
        
        # Home base (right side)
        self.ownership[:, GRID_WIDTH-HOME_BASE_WIDTH:] = 2
    
    def place_unit(self, x, y, unit_type, side, health=None):
        """
        Place a unit on the battlefield.
        
        Args:
            x: X coordinate (0-39)
            y: Y coordinate (0-24)
            unit_type: Type of unit to place
            side: 'ENEMY' or 'HOME'
            health: Optional health value (default to max health for unit type)
        
        Returns:
            True if successful, False if invalid placement
        """
        # Validate unit type
        if unit_type not in UNIT_STATS:
            print(f"Invalid unit type: {unit_type}")
            return False
        
        # Validate side
        if side not in SIDES:
            print(f"Invalid side: {side}")
            return False
        
        # Validate coordinates
        if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
            print(f"Invalid coordinates: ({x}, {y})")
            return False
        
        # Check if in correct base region
        side_idx = SIDES.index(side) + 1  # 1 for enemy, 2 for home
        if self.ownership[y, x] != side_idx:
            print(f"Cannot place {side} unit at ({x}, {y}) - wrong base region")
            return False
        
        # Check if cell is already occupied
        if np.any(self.grid[y, x] > 0):
            print(f"Cell ({x}, {y}) is already occupied")
            return False
        
        # Place unit
        unit_idx = UNIT_TO_IDX[unit_type]
        unit_health = health if health is not None else UNIT_STATS[unit_type]["health"]
        self.grid[y, x, unit_idx] = unit_health
        
        return True
    
    def get_unit_at(self, x, y):
        """
        Get information about a unit at the given coordinates.
        
        Returns:
            Tuple of (unit_type, health, side) or (None, 0, None) if no unit
        """
        if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
            return None, 0, None
        
        # Check if any unit exists at this position
        unit_values = self.grid[y, x]
        if not np.any(unit_values > 0):
            return None, 0, None
        
        # Find the unit type index
        unit_idx = np.argmax(unit_values)
        unit_type = UNIT_TYPES[unit_idx]
        health = unit_values[unit_idx]
        
        # Determine side
        side_idx = self.ownership[y, x]
        side = SIDES[side_idx - 1] if side_idx > 0 else None
        
        return unit_type, health, side
    
    def remove_unit(self, x, y):
        """Remove a unit from the specified location."""
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            self.grid[y, x] = 0
    
    def apply_formation(self, formation, side):
        """
        Apply a formation to the battlefield.
        
        Args:
            formation: 3D numpy array of shape (height, width, unit_types)
            side: 'ENEMY' or 'HOME'
        
        Returns:
            True if successful, False if invalid
        """
        if side not in SIDES:
            print(f"Invalid side: {side}")
            return False
        
        # Determine the base region to apply the formation
        if side == "ENEMY":
            base_x = 0
            base_width = ENEMY_BASE_WIDTH
        else:  # HOME
            base_x = GRID_WIDTH - HOME_BASE_WIDTH
            base_width = HOME_BASE_WIDTH
        
        # Check formation dimensions
        if formation.shape[0] != GRID_HEIGHT or formation.shape[1] != base_width:
            print(f"Invalid formation dimensions: {formation.shape}")
            return False
        
        # Apply formation by placing units
        for y in range(GRID_HEIGHT):
            for x in range(base_width):
                grid_x = base_x + x
                
                # Check each unit type channel
                for unit_idx, unit_type in enumerate(UNIT_TYPES):
                    health = formation[y, x, unit_idx]
                    if health > 0:
                        self.place_unit(grid_x, y, unit_type, side, health)
        
        return True
    
    def get_formation(self, side):
        """
        Extract the current formation for a given side.
        
        Args:
            side: 'ENEMY' or 'HOME'
        
        Returns:
            3D numpy array representing the formation
        """
        if side not in SIDES:
            print(f"Invalid side: {side}")
            return None
        
        # Determine the base region
        if side == "ENEMY":
            base_x = 0
            base_width = ENEMY_BASE_WIDTH
        else:  # HOME
            base_x = GRID_WIDTH - HOME_BASE_WIDTH
            base_width = HOME_BASE_WIDTH
        
        # Extract the formation
        formation = np.zeros((GRID_HEIGHT, base_width, len(UNIT_TYPES)))
        for y in range(GRID_HEIGHT):
            for x in range(base_width):
                grid_x = base_x + x
                formation[y, x] = self.grid[y, grid_x]
        
        return formation
    
    def move_units(self):
        """
        Move all units according to their movement rules.
        
        Returns:
            Number of units that moved
        """
        # Create a copy of the grid to read from while we modify the original
        old_grid = np.copy(self.grid)
        old_ownership = np.copy(self.ownership)
        
        # Track number of units moved
        units_moved = 0
        
        # Process each cell
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                # Skip empty cells
                if not np.any(old_grid[y, x] > 0):
                    continue
                
                # Find the unit type
                unit_idx = np.argmax(old_grid[y, x])
                unit_type = UNIT_TYPES[unit_idx]
                unit_health = old_grid[y, x, unit_idx]
                
                # Skip if unit is already dead
                if unit_health <= 0:
                    continue
                
                # Get the unit's side
                side_idx = old_ownership[y, x]
                if side_idx == 0:
                    continue  # Skip units with no owner
                
                side = SIDES[side_idx - 1]
                
                # Get movement speed for this unit type
                speed = UNIT_STATS[unit_type]["speed"]
                
                # Skip static units
                if speed == 0:
                    continue
                
                # Determine movement direction based on side
                direction = 1 if side == "ENEMY" else -1
                
                # Calculate new position
                new_x = x + (direction * speed)
                
                # Keep within grid boundaries
                new_x = max(0, min(new_x, GRID_WIDTH - 1))
                
                # Check if movement is possible
                if np.any(self.grid[y, new_x] > 0):
                    continue  # Destination is occupied
                
                # Move the unit
                self.grid[y, new_x, unit_idx] = unit_health
                self.grid[y, x, unit_idx] = 0
                self.ownership[y, new_x] = side_idx
                
                # If the original spot now has no units, clear ownership
                if not np.any(self.grid[y, x] > 0):
                    self.ownership[y, x] = 0
                
                units_moved += 1
        
        return units_moved
    
    def resolve_combat(self):
        """
        Resolve combat between adjacent units.
        
        Returns:
            Number of combat interactions
        """
        combat_count = 0
        
        # Create a damage matrix to store pending damage
        damage = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(UNIT_TYPES)))
        
        # Process each cell to calculate damage
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                # Skip empty cells
                if not np.any(self.grid[y, x] > 0):
                    continue
                
                # Find the unit
                unit_idx = np.argmax(self.grid[y, x])
                unit_type = UNIT_TYPES[unit_idx]
                unit_health = self.grid[y, x, unit_idx]
                
                # Skip if unit is already dead
                if unit_health <= 0:
                    continue
                
                # Get the unit's side
                side_idx = self.ownership[y, x]
                if side_idx == 0:
                    continue
                
                # Check adjacent cells for enemies
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    adj_y, adj_x = y + dy, x + dx
                    
                    # Skip if outside grid
                    if not (0 <= adj_x < GRID_WIDTH and 0 <= adj_y < GRID_HEIGHT):
                        continue
                    
                    # Skip if empty
                    if not np.any(self.grid[adj_y, adj_x] > 0):
                        continue
                    
                    # Check if enemy
                    adj_side_idx = self.ownership[adj_y, adj_x]
                    if adj_side_idx == 0 or adj_side_idx == side_idx:
                        continue  # Same side or no owner
                    
                    # Found an enemy! Apply damage
                    enemy_unit_idx = np.argmax(self.grid[adj_y, adj_x])
                    attack_power = UNIT_STATS[unit_type]["attack"]
                    
                    # Add damage to the damage matrix
                    damage[adj_y, adj_x, enemy_unit_idx] += attack_power
                    combat_count += 1
        
        # Apply all calculated damage
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                for unit_idx in range(len(UNIT_TYPES)):
                    if damage[y, x, unit_idx] > 0 and self.grid[y, x, unit_idx] > 0:
                        # Apply damage
                        self.grid[y, x, unit_idx] -= damage[y, x, unit_idx]
                        
                        # If unit is defeated, remove it
                        if self.grid[y, x, unit_idx] <= 0:
                            self.grid[y, x, unit_idx] = 0
                            
                            # If no units left in cell, clear ownership
                            if not np.any(self.grid[y, x] > 0):
                                self.ownership[y, x] = 0
        
        return combat_count
    
    def is_battle_complete(self, max_turns=100):
        """
        Check if the battle is complete.
        
        A battle is complete when:
        1. All units from one side are eliminated
        2. A maximum number of turns have elapsed
        
        Returns:
            True if battle is complete, False otherwise
        """
        # Check if either side has no units left
        enemy_health = self.calculate_total_health("ENEMY")
        home_health = self.calculate_total_health("HOME")
        
        if enemy_health == 0 or home_health == 0:
            return True
        
        # Check for maximum turns (not implemented yet)
        # This could be added as a property of the battlefield class
        
        return False
    
    def calculate_total_health(self, side):
        """
        Calculate the total remaining health for all units on a side.
        
        Args:
            side: 'ENEMY' or 'HOME'
            
        Returns:
            Total health value
        """
        if side not in SIDES:
            print(f"Invalid side: {side}")
            return 0
        
        side_idx = SIDES.index(side) + 1  # 1 for enemy, 2 for home
        
        # Create a mask for the specified side
        side_mask = (self.ownership == side_idx)
        
        # Sum the health of all units belonging to this side
        total_health = 0
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if side_mask[y, x]:
                    total_health += np.sum(self.grid[y, x])
        
        return total_health
    
    def clone(self):
        """Create a deep copy of the battlefield."""
        new_battlefield = Battlefield()
        new_battlefield.grid = np.copy(self.grid)
        new_battlefield.ownership = np.copy(self.ownership)
        return new_battlefield
    
    def summarize_state(self):
        """Generate a text summary of the battlefield state."""
        enemy_health = self.calculate_total_health("ENEMY")
        home_health = self.calculate_total_health("HOME")
        
        # Count units by type for each side
        enemy_units = {unit_type: 0 for unit_type in UNIT_TYPES}
        home_units = {unit_type: 0 for unit_type in UNIT_TYPES}
        
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                for unit_idx, unit_type in enumerate(UNIT_TYPES):
                    if self.grid[y, x, unit_idx] > 0:
                        side_idx = self.ownership[y, x]
                        if side_idx == 1:  # Enemy
                            enemy_units[unit_type] += 1
                        elif side_idx == 2:  # Home
                            home_units[unit_type] += 1
        
        summary = "Battlefield Summary:\n"
        summary += f"ENEMY: Total Health = {enemy_health}\n"
        for unit_type, count in enemy_units.items():
            if count > 0:
                summary += f"  - {unit_type}: {count} units\n"
        
        summary += f"HOME: Total Health = {home_health}\n"
        for unit_type, count in home_units.items():
            if count > 0:
                summary += f"  - {unit_type}: {count} units\n"
        
        return summary 