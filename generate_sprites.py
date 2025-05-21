"""
Generate sprites for all unit types.
"""

import os
import pygame
from src.utils.constants import UNIT_TYPES, UNIT_COLORS, GRID_CELL_SIZE

def main():
    """Generate sprite images for all unit types."""
    # Initialize pygame
    pygame.init()
    
    # Create sprites directory if it doesn't exist
    if not os.path.exists("sprites"):
        os.makedirs("sprites")
    
    # Create a sprite for each unit type
    for unit_type in UNIT_TYPES:
        # Get color for this unit type
        color = UNIT_COLORS.get(unit_type, (200, 200, 200))
        
        # Create surface
        surface = pygame.Surface((GRID_CELL_SIZE, GRID_CELL_SIZE))
        
        # Fill with color
        pygame.draw.rect(surface, color, (0, 0, GRID_CELL_SIZE, GRID_CELL_SIZE))
        
        # Add border
        pygame.draw.rect(surface, (0, 0, 0), (0, 0, GRID_CELL_SIZE, GRID_CELL_SIZE), 2)
        
        # Add unit-specific details
        if unit_type == "SOLDIER":
            # Simple soldier icon (person shape)
            pygame.draw.circle(surface, (0, 0, 0), (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 3), GRID_CELL_SIZE // 6)
            pygame.draw.line(surface, (0, 0, 0), 
                            (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 3 + 2), 
                            (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 3 * 2), 
                            2)
            pygame.draw.line(surface, (0, 0, 0),
                            (GRID_CELL_SIZE // 3, GRID_CELL_SIZE // 2),
                            (GRID_CELL_SIZE // 3 * 2, GRID_CELL_SIZE // 2),
                            2)
            pygame.draw.line(surface, (0, 0, 0),
                            (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 3 * 2),
                            (GRID_CELL_SIZE // 3, GRID_CELL_SIZE // 5 * 4),
                            2)
            pygame.draw.line(surface, (0, 0, 0),
                            (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 3 * 2),
                            (GRID_CELL_SIZE // 3 * 2, GRID_CELL_SIZE // 5 * 4),
                            2)
        
        elif unit_type == "TANK":
            # Tank shape
            pygame.draw.rect(surface, (0, 0, 0), 
                            (GRID_CELL_SIZE // 4, GRID_CELL_SIZE // 3, 
                            GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 2), 
                            2)
            pygame.draw.rect(surface, (0, 0, 0),
                            (GRID_CELL_SIZE // 3, GRID_CELL_SIZE // 4,
                            GRID_CELL_SIZE // 3, GRID_CELL_SIZE // 8),
                            0)  # Gun barrel
            pygame.draw.circle(surface, (0, 0, 0), 
                              (GRID_CELL_SIZE // 3, GRID_CELL_SIZE // 5 * 4), 
                              GRID_CELL_SIZE // 8)  # Wheel
            pygame.draw.circle(surface, (0, 0, 0), 
                              (GRID_CELL_SIZE // 3 * 2, GRID_CELL_SIZE // 5 * 4), 
                              GRID_CELL_SIZE // 8)  # Wheel
        
        elif unit_type == "LANDMINE":
            # Landmine shape
            pygame.draw.circle(surface, (0, 0, 0), 
                              (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 2), 
                              GRID_CELL_SIZE // 3, 2)
            pygame.draw.line(surface, (0, 0, 0),
                            (GRID_CELL_SIZE // 3, GRID_CELL_SIZE // 3),
                            (GRID_CELL_SIZE // 3 * 2, GRID_CELL_SIZE // 3 * 2),
                            2)
            pygame.draw.line(surface, (0, 0, 0),
                            (GRID_CELL_SIZE // 3 * 2, GRID_CELL_SIZE // 3),
                            (GRID_CELL_SIZE // 3, GRID_CELL_SIZE // 3 * 2),
                            2)
        
        elif unit_type == "FIGHTER_JET":
            # Jet shape
            points = [
                (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 4),  # Nose
                (GRID_CELL_SIZE // 4, GRID_CELL_SIZE // 2),  # Left wing
                (GRID_CELL_SIZE // 3, GRID_CELL_SIZE // 3 * 2),  # Left tail
                (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 3 * 2),  # Tail center
                (GRID_CELL_SIZE // 3 * 2, GRID_CELL_SIZE // 3 * 2),  # Right tail
                (GRID_CELL_SIZE // 4 * 3, GRID_CELL_SIZE // 2),  # Right wing
            ]
            pygame.draw.polygon(surface, (0, 0, 0), points)
        
        elif unit_type == "SHIELDED_SOLDIER":
            # Soldier with shield
            pygame.draw.circle(surface, (0, 0, 0), (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 3), GRID_CELL_SIZE // 6)
            pygame.draw.line(surface, (0, 0, 0), 
                            (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 3 + 2), 
                            (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 3 * 2), 
                            2)
            pygame.draw.arc(surface, (0, 0, 0),
                           (GRID_CELL_SIZE // 4, GRID_CELL_SIZE // 3,
                           GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 2),
                           0, 3.14, 2)  # Shield
        
        elif unit_type == "ARTILLERY":
            # Artillery shape
            pygame.draw.rect(surface, (0, 0, 0), 
                           (GRID_CELL_SIZE // 4, GRID_CELL_SIZE // 2, 
                           GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 3), 
                           2)  # Base
            pygame.draw.line(surface, (0, 0, 0),
                           (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 3),
                           (GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 5 * 4),
                           4)  # Gun barrel
            pygame.draw.circle(surface, (0, 0, 0), 
                             (GRID_CELL_SIZE // 3, GRID_CELL_SIZE // 4 * 3), 
                             GRID_CELL_SIZE // 8)  # Wheel
            pygame.draw.circle(surface, (0, 0, 0), 
                             (GRID_CELL_SIZE // 3 * 2, GRID_CELL_SIZE // 4 * 3), 
                             GRID_CELL_SIZE // 8)  # Wheel
        
        elif unit_type == "GUARD_TOWER":
            # Tower shape
            pygame.draw.rect(surface, (0, 0, 0), 
                           (GRID_CELL_SIZE // 3, GRID_CELL_SIZE // 4, 
                           GRID_CELL_SIZE // 3, GRID_CELL_SIZE // 2), 
                           2)  # Tower body
            pygame.draw.rect(surface, (0, 0, 0),
                           (GRID_CELL_SIZE // 4, GRID_CELL_SIZE // 4,
                           GRID_CELL_SIZE // 2, GRID_CELL_SIZE // 5),
                           2)  # Tower top
        
        # Save the sprite
        pygame.image.save(surface, f"sprites/{unit_type.lower()}.png")
        print(f"Created {unit_type} sprite")
    
    pygame.quit()
    print("All sprites generated")

if __name__ == "__main__":
    main() 