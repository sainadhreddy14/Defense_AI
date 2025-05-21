"""
Standard military formation patterns.
"""

import numpy as np
from ..utils.constants import UNIT_TYPES, UNIT_TO_IDX, UNIT_STATS, GRID_HEIGHT


def create_line_formation(unit_mix, width=10, height=GRID_HEIGHT):
    """
    Create a basic line formation across the front.
    
    Args:
        unit_mix: Dictionary mapping unit types to counts
        width: Width of the formation (default 10)
        height: Height of the formation (default 25)
    
    Returns:
        3D numpy array representing the formation
    """
    # Create empty formation
    formation = np.zeros((height, width, len(UNIT_TYPES)))
    
    # Determine rows for different unit types
    # We place units in rows by type
    total_units = sum(unit_mix.values())
    
    # Place the units row by row
    placed_units = {unit_type: 0 for unit_type in UNIT_TYPES}
    
    # First pass: place the most valuable/powerful units
    priority_units = ["TANK", "FIGHTER_JET", "ARTILLERY", "GUARD_TOWER"]
    
    # Start from the center rows and work outward
    center_row = height // 2
    row_order = [center_row]
    for i in range(1, height):
        if center_row - i >= 0:
            row_order.append(center_row - i)
        if center_row + i < height:
            row_order.append(center_row + i)
    
    # First pass: place priority units
    for row in row_order:
        for unit_type in priority_units:
            # Skip if no units of this type remain
            if placed_units[unit_type] >= unit_mix.get(unit_type, 0):
                continue
            
            # Try to place the unit in this row
            unit_idx = UNIT_TO_IDX[unit_type]
            for col in range(width):
                # If cell is empty and we have units left
                if np.all(formation[row, col] == 0) and placed_units[unit_type] < unit_mix.get(unit_type, 0):
                    # Place unit
                    formation[row, col, unit_idx] = UNIT_STATS[unit_type]["health"]
                    placed_units[unit_type] += 1
    
    # Second pass: place remaining units
    for row in row_order:
        for unit_type in UNIT_TYPES:
            if unit_type in priority_units:
                continue  # Skip priority units handled above
                
            # Skip if no units of this type remain
            if placed_units[unit_type] >= unit_mix.get(unit_type, 0):
                continue
            
            # Try to place the unit in this row
            unit_idx = UNIT_TO_IDX[unit_type]
            for col in range(width):
                # If cell is empty and we have units left
                if np.all(formation[row, col] == 0) and placed_units[unit_type] < unit_mix.get(unit_type, 0):
                    # Place unit
                    formation[row, col, unit_idx] = UNIT_STATS[unit_type]["health"]
                    placed_units[unit_type] += 1
    
    return formation


def create_wedge_formation(unit_mix, width=10, height=GRID_HEIGHT):
    """
    Create a wedge (triangle) formation for breakthrough.
    
    Args:
        unit_mix: Dictionary mapping unit types to counts
        width: Width of the formation (default 10)
        height: Height of the formation (default 25)
    
    Returns:
        3D numpy array representing the formation
    """
    # Create empty formation
    formation = np.zeros((height, width, len(UNIT_TYPES)))
    
    # Calculate the center of the formation
    center_row = height // 2
    center_col = width // 2
    
    # Create a mask for wedge shape
    wedge_mask = np.zeros((height, width), dtype=bool)
    
    # Generate wedge shape
    for row in range(height):
        row_distance = abs(row - center_row)
        # Width of wedge at this row
        wedge_width = max(1, width - row_distance * 2)
        
        # Calculate start column for this row
        start_col = center_col - wedge_width // 2
        
        # Ensure start column is within bounds
        start_col = max(0, min(start_col, width - 1))
        
        # Set columns in this row as part of the wedge
        for col in range(start_col, min(start_col + wedge_width, width)):
            wedge_mask[row, col] = True
    
    # Place the strongest units at the tip (front) of the wedge
    placed_units = {unit_type: 0 for unit_type in UNIT_TYPES}
    
    # Priority units for the tip of the wedge
    tip_units = ["TANK", "FIGHTER_JET", "ARTILLERY"]
    
    # Calculate the front rows of the wedge (1/3 of the rows)
    front_rows = list(range(height // 3, 2 * height // 3))
    
    # Place tip units at the front of the wedge
    for row in front_rows:
        for col in range(width):
            if not wedge_mask[row, col]:
                continue  # Skip if not part of wedge
                
            # Try each tip unit type
            for unit_type in tip_units:
                if placed_units[unit_type] >= unit_mix.get(unit_type, 0):
                    continue  # Skip if no units of this type left
                
                # Place unit
                unit_idx = UNIT_TO_IDX[unit_type]
                formation[row, col, unit_idx] = UNIT_STATS[unit_type]["health"]
                placed_units[unit_type] += 1
                break  # Move to next cell
    
    # Place remaining units in the rest of the wedge
    for row in range(height):
        for col in range(width):
            if not wedge_mask[row, col] or np.any(formation[row, col] > 0):
                continue  # Skip if not part of wedge or already filled
            
            # Try each remaining unit type
            for unit_type in UNIT_TYPES:
                if placed_units[unit_type] >= unit_mix.get(unit_type, 0):
                    continue  # Skip if no units of this type left
                
                # Place unit
                unit_idx = UNIT_TO_IDX[unit_type]
                formation[row, col, unit_idx] = UNIT_STATS[unit_type]["health"]
                placed_units[unit_type] += 1
                break  # Move to next cell
    
    return formation


def create_echelon_formation(unit_mix, width=10, height=GRID_HEIGHT, right_sided=True):
    """
    Create an echelon formation (diagonal line) for flanking.
    
    Args:
        unit_mix: Dictionary mapping unit types to counts
        width: Width of the formation (default 10)
        height: Height of the formation (default 25)
        right_sided: If True, echelon runs from top-left to bottom-right,
                    if False, from bottom-left to top-right
    
    Returns:
        3D numpy array representing the formation
    """
    # Create empty formation
    formation = np.zeros((height, width, len(UNIT_TYPES)))
    
    # Create echelon mask
    echelon_mask = np.zeros((height, width), dtype=bool)
    
    # Calculate the angle of the echelon
    # For a 10x25 grid, one column shift every ~2.5 rows
    row_step = height / width
    
    # Generate echelon shape
    for col in range(width):
        # Calculate which rows this column occupies
        if right_sided:
            # Top-left to bottom-right
            start_row = int(col * row_step)
            end_row = start_row + max(3, height // 8)  # Make the echelon thicker
        else:
            # Bottom-left to top-right
            start_row = height - 1 - int(col * row_step)
            end_row = start_row - max(3, height // 8)  # Make the echelon thicker
            # Ensure end_row is not negative
            end_row = max(0, end_row)
        
        # Mark cells in echelon
        if right_sided:
            for row in range(start_row, min(end_row, height)):
                echelon_mask[row, col] = True
        else:
            for row in range(end_row, min(start_row + 1, height)):
                echelon_mask[row, col] = True
    
    # Place units along the echelon
    placed_units = {unit_type: 0 for unit_type in UNIT_TYPES}
    
    # Priority for mobile units
    mobile_units = ["TANK", "FIGHTER_JET", "SOLDIER", "SHIELDED_SOLDIER"]
    
    # First pass: place mobile units
    for row in range(height):
        for col in range(width):
            if not echelon_mask[row, col]:
                continue  # Skip if not part of echelon
                
            # Try each mobile unit type
            for unit_type in mobile_units:
                if placed_units[unit_type] >= unit_mix.get(unit_type, 0):
                    continue  # Skip if no units of this type left
                
                # Place unit
                unit_idx = UNIT_TO_IDX[unit_type]
                formation[row, col, unit_idx] = UNIT_STATS[unit_type]["health"]
                placed_units[unit_type] += 1
                break  # Move to next cell
    
    # Second pass: place remaining units
    for row in range(height):
        for col in range(width):
            if not echelon_mask[row, col] or np.any(formation[row, col] > 0):
                continue  # Skip if not part of echelon or already filled
            
            # Try each remaining unit type
            for unit_type in UNIT_TYPES:
                if unit_type in mobile_units:
                    continue  # Skip mobile units already placed
                
                if placed_units[unit_type] >= unit_mix.get(unit_type, 0):
                    continue  # Skip if no units of this type left
                
                # Place unit
                unit_idx = UNIT_TO_IDX[unit_type]
                formation[row, col, unit_idx] = UNIT_STATS[unit_type]["health"]
                placed_units[unit_type] += 1
                break  # Move to next cell
    
    return formation


def create_refused_flank_formation(unit_mix, width=10, height=GRID_HEIGHT, strong_side="right"):
    """
    Create a formation with one strong flank and one refused (weaker) flank.
    
    Args:
        unit_mix: Dictionary mapping unit types to counts
        width: Width of the formation (default 10)
        height: Height of the formation (default 25)
        strong_side: Which flank is strong ("right", "left", "top", "bottom")
    
    Returns:
        3D numpy array representing the formation
    """
    # Create empty formation
    formation = np.zeros((height, width, len(UNIT_TYPES)))
    
    # Create masks for strong and weak areas
    strong_mask = np.zeros((height, width), dtype=bool)
    weak_mask = np.zeros((height, width), dtype=bool)
    
    # Set up masks based on strong side
    if strong_side == "right":
        strong_mask[:, width//2:] = True
        weak_mask[:, :width//2] = True
    elif strong_side == "left":
        strong_mask[:, :width//2] = True
        weak_mask[:, width//2:] = True
    elif strong_side == "top":
        strong_mask[:height//2, :] = True
        weak_mask[height//2:, :] = True
    elif strong_side == "bottom":
        strong_mask[height//2:, :] = True
        weak_mask[:height//2, :] = True
    else:
        # Default to right flank if invalid parameter
        strong_mask[:, width//2:] = True
        weak_mask[:, :width//2] = True
    
    # Allocate units between strong and weak flanks
    # Strong flank gets 70% of units
    strong_units = {}
    weak_units = {}
    
    for unit_type, count in unit_mix.items():
        strong_count = int(count * 0.7)
        weak_count = count - strong_count
        
        strong_units[unit_type] = strong_count
        weak_units[unit_type] = weak_count
    
    # Place units on strong flank
    placed_strong = {unit_type: 0 for unit_type in UNIT_TYPES}
    
    # First pass: place strong flank priority units (tanks, artillery, fighters)
    priority_units = ["TANK", "ARTILLERY", "FIGHTER_JET"]
    
    for row in range(height):
        for col in range(width):
            if not strong_mask[row, col]:
                continue  # Skip if not in strong flank
                
            # Try each priority unit type
            for unit_type in priority_units:
                if placed_strong[unit_type] >= strong_units.get(unit_type, 0):
                    continue  # Skip if no units of this type left
                
                # Place unit
                unit_idx = UNIT_TO_IDX[unit_type]
                formation[row, col, unit_idx] = UNIT_STATS[unit_type]["health"]
                placed_strong[unit_type] += 1
                break  # Move to next cell
    
    # Second pass: place remaining strong flank units
    for row in range(height):
        for col in range(width):
            if not strong_mask[row, col] or np.any(formation[row, col] > 0):
                continue  # Skip if not in strong flank or already filled
            
            # Try each remaining unit type
            for unit_type in UNIT_TYPES:
                if unit_type in priority_units:
                    continue  # Skip priority units already placed
                
                if placed_strong[unit_type] >= strong_units.get(unit_type, 0):
                    continue  # Skip if no units of this type left
                
                # Place unit
                unit_idx = UNIT_TO_IDX[unit_type]
                formation[row, col, unit_idx] = UNIT_STATS[unit_type]["health"]
                placed_strong[unit_type] += 1
                break  # Move to next cell
    
    # Place units on weak flank
    placed_weak = {unit_type: 0 for unit_type in UNIT_TYPES}
    
    # Weak flank prioritizes defensive units
    defensive_units = ["GUARD_TOWER", "LANDMINE", "SHIELDED_SOLDIER"]
    
    # First pass: place weak flank defensive units
    for row in range(height):
        for col in range(width):
            if not weak_mask[row, col]:
                continue  # Skip if not in weak flank
                
            # Try each defensive unit type
            for unit_type in defensive_units:
                if placed_weak[unit_type] >= weak_units.get(unit_type, 0):
                    continue  # Skip if no units of this type left
                
                # Place unit
                unit_idx = UNIT_TO_IDX[unit_type]
                formation[row, col, unit_idx] = UNIT_STATS[unit_type]["health"]
                placed_weak[unit_type] += 1
                break  # Move to next cell
    
    # Second pass: place remaining weak flank units
    for row in range(height):
        for col in range(width):
            if not weak_mask[row, col] or np.any(formation[row, col] > 0):
                continue  # Skip if not in weak flank or already filled
            
            # Try each remaining unit type
            for unit_type in UNIT_TYPES:
                if unit_type in defensive_units:
                    continue  # Skip defensive units already placed
                
                if placed_weak[unit_type] >= weak_units.get(unit_type, 0):
                    continue  # Skip if no units of this type left
                
                # Place unit
                unit_idx = UNIT_TO_IDX[unit_type]
                formation[row, col, unit_idx] = UNIT_STATS[unit_type]["health"]
                placed_weak[unit_type] += 1
                break  # Move to next cell
    
    return formation


def create_balanced_formation(unit_mix, width=10, height=GRID_HEIGHT):
    """
    Create a balanced formation with uniform distribution.
    
    Args:
        unit_mix: Dictionary mapping unit types to counts
        width: Width of the formation (default 10)
        height: Height of the formation (default 25)
    
    Returns:
        3D numpy array representing the formation
    """
    # Create empty formation
    formation = np.zeros((height, width, len(UNIT_TYPES)))
    
    # Create a list of all cells
    cells = [(row, col) for row in range(height) for col in range(width)]
    
    # Shuffle cells for random placement
    np.random.shuffle(cells)
    
    # Place units
    placed_units = {unit_type: 0 for unit_type in UNIT_TYPES}
    
    for row, col in cells:
        # Skip if cell already filled
        if np.any(formation[row, col] > 0):
            continue
        
        # Try to place a unit
        for unit_type in UNIT_TYPES:
            if placed_units[unit_type] >= unit_mix.get(unit_type, 0):
                continue  # Skip if no units of this type left
            
            # Place unit
            unit_idx = UNIT_TO_IDX[unit_type]
            formation[row, col, unit_idx] = UNIT_STATS[unit_type]["health"]
            placed_units[unit_type] += 1
            break  # Move to next cell
    
    return formation 