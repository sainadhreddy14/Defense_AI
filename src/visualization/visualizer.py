"""
Visualization module for rendering the battlefield.
"""

import pygame
import numpy as np
import time
import os
from ..utils.constants import (
    GRID_WIDTH, GRID_HEIGHT, ENEMY_BASE_WIDTH, HOME_BASE_WIDTH,
    GRID_CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT,
    COLORS, UNIT_COLORS, UNIT_TYPES, UNIT_STATS
)


class BattlefieldVisualizer:
    """Visualizes the battlefield and battle simulations using Pygame."""
    
    def __init__(self):
        """Initialize the visualization system."""
        pygame.init()
        print("Pygame initialized!")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        print(f"Display set up with size: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        pygame.display.set_caption("Battleground Simulator")
        
        # Font for text
        self.font = pygame.font.SysFont("Arial", 12)
        self.header_font = pygame.font.SysFont("Arial", 16, bold=True)
        
        # Load unit sprites if available
        self.sprites = {}
        self.load_sprites()
        print(f"Loaded {len(self.sprites)} sprites")
    
    def load_sprites(self):
        """Load sprite images for units."""
        sprites_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sprites")
        print(f"Loading sprites from: {sprites_dir}")
        
        # Try to load sprites for each unit type
        for unit_type in UNIT_TYPES:
            sprite_path = os.path.join(sprites_dir, f"{unit_type.lower()}.png")
            try:
                if os.path.exists(sprite_path):
                    print(f"Loading sprite: {sprite_path}")
                    self.sprites[unit_type] = pygame.image.load(sprite_path)
                    self.sprites[unit_type] = pygame.transform.scale(
                        self.sprites[unit_type], 
                        (GRID_CELL_SIZE, GRID_CELL_SIZE)
                    )
                else:
                    print(f"Sprite file not found: {sprite_path}")
            except pygame.error as e:
                print(f"Could not load sprite: {sprite_path}, Error: {e}")
    
    def render_battlefield(self, battlefield):
        """
        Render the current state of the battlefield.
        
        Args:
            battlefield: Battlefield object to render
        """
        # Print some debug info
        print("Rendering battlefield")
        enemy_health = battlefield.calculate_total_health("ENEMY")
        home_health = battlefield.calculate_total_health("HOME")
        print(f"Enemy health: {enemy_health}, Home health: {home_health}")
        
        # Clear screen
        self.screen.fill(COLORS["WHITE"])
        
        # Draw grid
        self._draw_grid()
        
        # Draw base regions
        self._draw_base_regions()
        
        # Draw units
        units_drawn = self._draw_units(battlefield)
        print(f"Drew {units_drawn} units on the battlefield")
        
        # Draw UI elements
        self._draw_ui(battlefield)
        
        # Update display
        pygame.display.flip()
        pygame.time.delay(100)  # Add a small delay to ensure rendering completes
        print("Battlefield rendered and display updated")
    
    def _draw_grid(self):
        """Draw the battlefield grid."""
        # Draw horizontal grid lines
        for y in range(GRID_HEIGHT + 1):
            pygame.draw.line(
                self.screen,
                COLORS["GRID_LINE"],
                (0, y * GRID_CELL_SIZE),
                (SCREEN_WIDTH, y * GRID_CELL_SIZE),
                1
            )
        
        # Draw vertical grid lines
        for x in range(GRID_WIDTH + 1):
            pygame.draw.line(
                self.screen,
                COLORS["GRID_LINE"],
                (x * GRID_CELL_SIZE, 0),
                (x * GRID_CELL_SIZE, SCREEN_HEIGHT),
                1
            )
    
    def _draw_base_regions(self):
        """Draw the enemy and home base regions."""
        # Enemy base (translucent red)
        enemy_base_surface = pygame.Surface((ENEMY_BASE_WIDTH * GRID_CELL_SIZE, SCREEN_HEIGHT), pygame.SRCALPHA)
        enemy_base_surface.fill((255, 100, 100, 50))  # Semi-transparent red
        self.screen.blit(enemy_base_surface, (0, 0))
        
        # Home base (translucent blue)
        home_base_surface = pygame.Surface((HOME_BASE_WIDTH * GRID_CELL_SIZE, SCREEN_HEIGHT), pygame.SRCALPHA)
        home_base_surface.fill((100, 100, 255, 50))  # Semi-transparent blue
        self.screen.blit(home_base_surface, (SCREEN_WIDTH - HOME_BASE_WIDTH * GRID_CELL_SIZE, 0))
    
    def _draw_units(self, battlefield):
        """
        Draw all units on the battlefield.
        
        Args:
            battlefield: Battlefield object containing unit data
            
        Returns:
            Number of units drawn
        """
        units_drawn = 0
        
        # Draw each unit
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                unit_type, health, side = battlefield.get_unit_at(x, y)
                
                if unit_type is not None:
                    units_drawn += 1
                    
                    # Position for the unit
                    pos_x = x * GRID_CELL_SIZE
                    pos_y = y * GRID_CELL_SIZE
                    
                    # Determine unit color based on side
                    if side == "ENEMY":
                        border_color = COLORS["ENEMY_COLOR"]
                    else:  # HOME
                        border_color = COLORS["HOME_COLOR"]
                    
                    # Draw the unit
                    if unit_type in self.sprites:
                        # Use sprite if available
                        self.screen.blit(self.sprites[unit_type], (pos_x, pos_y))
                        
                        # Draw border to indicate side
                        pygame.draw.rect(
                            self.screen,
                            border_color,
                            (pos_x, pos_y, GRID_CELL_SIZE, GRID_CELL_SIZE),
                            2
                        )
                    else:
                        # Draw colored rect if no sprite
                        unit_color = UNIT_COLORS.get(unit_type, COLORS["GREY"])
                        
                        # Draw filled rect
                        pygame.draw.rect(
                            self.screen,
                            unit_color,
                            (pos_x, pos_y, GRID_CELL_SIZE, GRID_CELL_SIZE)
                        )
                        
                        # Draw border to indicate side
                        pygame.draw.rect(
                            self.screen,
                            border_color,
                            (pos_x, pos_y, GRID_CELL_SIZE, GRID_CELL_SIZE),
                            2
                        )
                    
                    # Draw health indicator
                    max_health = UNIT_STATS[unit_type]["health"]
                    health_percent = health / max_health
                    
                    # Health bar background
                    pygame.draw.rect(
                        self.screen,
                        COLORS["BLACK"],
                        (pos_x + 2, pos_y + GRID_CELL_SIZE - 5, GRID_CELL_SIZE - 4, 3)
                    )
                    
                    # Health bar fill
                    health_width = int((GRID_CELL_SIZE - 4) * health_percent)
                    if health_percent > 0.6:
                        health_color = COLORS["GREEN"]
                    elif health_percent > 0.3:
                        health_color = COLORS["YELLOW"]
                    else:
                        health_color = COLORS["RED"]
                    
                    pygame.draw.rect(
                        self.screen,
                        health_color,
                        (pos_x + 2, pos_y + GRID_CELL_SIZE - 5, health_width, 3)
                    )
        
        return units_drawn
    
    def _draw_ui(self, battlefield):
        """
        Draw UI elements like health totals and unit counts.
        
        Args:
            battlefield: Battlefield object
        """
        # Calculate total health for each side
        enemy_health = battlefield.calculate_total_health("ENEMY")
        home_health = battlefield.calculate_total_health("HOME")
        
        # Draw health bars at the top
        # Enemy health bar (left)
        pygame.draw.rect(
            self.screen,
            COLORS["BLACK"],
            (10, 10, 200, 20),
            1
        )
        enemy_health_width = int(min(enemy_health / 5000, 1) * 198)
        pygame.draw.rect(
            self.screen,
            COLORS["RED"],
            (11, 11, enemy_health_width, 18)
        )
        enemy_health_text = self.font.render(f"Enemy: {int(enemy_health)}", True, COLORS["BLACK"])
        self.screen.blit(enemy_health_text, (15, 12))
        
        # Home health bar (right)
        pygame.draw.rect(
            self.screen,
            COLORS["BLACK"],
            (SCREEN_WIDTH - 210, 10, 200, 20),
            1
        )
        home_health_width = int(min(home_health / 5000, 1) * 198)
        pygame.draw.rect(
            self.screen,
            COLORS["BLUE"],
            (SCREEN_WIDTH - 209, 11, home_health_width, 18)
        )
        home_health_text = self.font.render(f"Home: {int(home_health)}", True, COLORS["BLACK"])
        self.screen.blit(home_health_text, (SCREEN_WIDTH - 205, 12))
    
    def render_battle_replay(self, battle_history, speed=0.5):
        """
        Render a step-by-step replay of a battle.
        
        Args:
            battle_history: List of Battlefield objects representing the battle history
            speed: Seconds between frames
        """
        print(f"Starting battle replay with {len(battle_history)} frames")
        
        # Display initial state
        self.render_battlefield(battle_history[0])
        
        # Display message to start
        start_text = self.header_font.render("Press any key to start replay", True, COLORS["BLACK"])
        self.screen.blit(start_text, (SCREEN_WIDTH // 2 - 100, 50))
        skip_text = self.font.render("Press 'S' anytime to skip to end", True, COLORS["BLACK"])
        self.screen.blit(skip_text, (SCREEN_WIDTH // 2 - 90, 70))
        pygame.display.flip()
        
        print("Waiting for user to start the replay...")
        try:
            # Wait for user to start the replay
            self._wait_for_key()
            print("User started the replay")
            
            # Flag to track if user wants to skip to the end
            skip_to_end = False
            
            # Render each state in sequence
            for i, battlefield in enumerate(battle_history[1:]):
                # Process events and check for skip key
                skip_to_end = self._check_for_skip_key()
                if skip_to_end:
                    print("User pressed skip key - jumping to end")
                    break
                
                # Render current state
                print(f"Rendering frame {i+1}/{len(battle_history)-1}")
                self.render_battlefield(battlefield)
                
                # Add turn number
                turn_text = self.header_font.render(f"Turn: {i//2 + 1}", True, COLORS["BLACK"])
                self.screen.blit(turn_text, (SCREEN_WIDTH // 2 - 40, 10))
                
                # Add phase info (movement or combat)
                phase = "Movement" if i % 2 == 0 else "Combat"
                phase_text = self.font.render(phase, True, COLORS["BLACK"])
                self.screen.blit(phase_text, (SCREEN_WIDTH // 2 - 25, 30))
                
                # Reminder about skipping
                skip_reminder = self.font.render("Press 'S' to skip to end", True, COLORS["BLACK"])
                self.screen.blit(skip_reminder, (SCREEN_WIDTH // 2 - 70, 50))
                
                pygame.display.flip()
                
                # Wait between frames
                time.sleep(speed)
            
            # If we broke out of the loop or finished it, show the final state
            if skip_to_end:
                print("Skipping to aftermath")
                self.render_battle_aftermath(battle_history)
            else:
                # Show final state
                self.render_battlefield(battle_history[-1])
                
                # Determine winner
                enemy_health = battle_history[-1].calculate_total_health("ENEMY")
                home_health = battle_history[-1].calculate_total_health("HOME")
                
                if enemy_health > home_health:
                    winner = "ENEMY"
                    winner_color = COLORS["RED"]
                elif home_health > enemy_health:
                    winner = "HOME"
                    winner_color = COLORS["BLUE"]
                else:
                    winner = "DRAW"
                    winner_color = COLORS["BLACK"]
                
                # Display winner
                winner_text = self.header_font.render(f"Winner: {winner}", True, winner_color)
                self.screen.blit(winner_text, (SCREEN_WIDTH // 2 - 50, 50))
                
                close_text = self.font.render("Press any key to continue", True, COLORS["BLACK"])
                self.screen.blit(close_text, (SCREEN_WIDTH // 2 - 70, 70))
                
                pygame.display.flip()
                
                # Wait for user to close
                print("Waiting for user to close battle replay...")
                self._wait_for_key()
                print("Battle replay finished")
        except Exception as e:
            print(f"Error during battle replay: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Make sure to keep processing events to avoid freezing
            self._process_events()
            
    def _check_for_skip_key(self):
        """
        Check if the user pressed the skip key ('S').
        
        Returns:
            True if the skip key was pressed, False otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                print("User closed the window")
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    return True
        
        return False
    
    def render_battle_aftermath(self, battle_history):
        """
        Render only the final state of a battle without step-by-step simulation.
        
        Args:
            battle_history: List of Battlefield objects representing the battle history
        """
        if not battle_history:
            print("No battle history to display")
            return
            
        print("Rendering battle aftermath")
        
        # Display final state
        self.render_battlefield(battle_history[-1])
        
        # Determine winner
        enemy_health = battle_history[-1].calculate_total_health("ENEMY")
        home_health = battle_history[-1].calculate_total_health("HOME")
        
        if enemy_health > home_health:
            winner = "ENEMY"
            winner_color = COLORS["RED"]
        elif home_health > enemy_health:
            winner = "HOME"
            winner_color = COLORS["BLUE"]
        else:
            winner = "DRAW"
            winner_color = COLORS["BLACK"]
        
        # Just display the winner
        winner_text = self.header_font.render(f"Winner: {winner}", True, winner_color)
        self.screen.blit(winner_text, (SCREEN_WIDTH // 2 - 50, 50))
        
        close_text = self.font.render("Press any key to continue", True, COLORS["BLACK"])
        self.screen.blit(close_text, (SCREEN_WIDTH // 2 - 70, 70))
        
        pygame.display.flip()
        
        # Wait for user to close
        print("Waiting for user to close battle aftermath...")
        self._wait_for_key()
        print("Battle aftermath closed")
    
    def _wait_for_key(self):
        """Wait for a key press."""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print("User closed the window")
                    return
                elif event.type == pygame.KEYDOWN:
                    waiting = False
            
            # Add a small delay to reduce CPU usage
            pygame.time.delay(50)
    
    def _process_events(self):
        """Process pygame events without waiting."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                print("User closed the window")
                return
    
    def close(self):
        """Close the visualization system."""
        pygame.quit() 