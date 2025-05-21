"""
Constants and configuration values for the battleground simulator.
"""

# Battlefield dimensions
GRID_WIDTH = 40
GRID_HEIGHT = 25
ENEMY_BASE_WIDTH = 10
HOME_BASE_WIDTH = 10
BATTLEGROUND_WIDTH = GRID_WIDTH - ENEMY_BASE_WIDTH - HOME_BASE_WIDTH

# Unit types
UNIT_TYPES = ["SOLDIER", "TANK", "LANDMINE", "FIGHTER_JET", 
              "SHIELDED_SOLDIER", "ARTILLERY", "GUARD_TOWER"]

# Map unit types to indices for the grid representation
UNIT_TO_IDX = {
    "SOLDIER": 0,
    "TANK": 1,
    "LANDMINE": 2,
    "FIGHTER_JET": 3,
    "SHIELDED_SOLDIER": 4,
    "ARTILLERY": 5,
    "GUARD_TOWER": 6
}

# Unit stats
UNIT_STATS = {
    "SOLDIER": {
        "attack": 200, 
        "health": 200, 
        "cost": 15, 
        "max": 50,
        "speed": 1,
        "description": "Basic infantry unit"
    },
    "TANK": {
        "attack": 350, 
        "health": 600, 
        "cost": 50, 
        "max": 20,
        "speed": 1,
        "description": "Heavy armored unit"
    },
    "LANDMINE": {
        "attack": 200, 
        "health": 50,  # Low health as it's single-use
        "cost": 10, 
        "max": 10,
        "speed": 0,  # Static
        "description": "Static defensive unit"
    },
    "FIGHTER_JET": {
        "attack": 700, 
        "health": 600, 
        "cost": 100, 
        "max": 7,
        "speed": 2,  # Faster movement
        "description": "Fast air unit with high attack"
    },
    "SHIELDED_SOLDIER": {
        "attack": 100, 
        "health": 400, 
        "cost": 20, 
        "max": 50,
        "speed": 1,
        "description": "Defensive infantry with high health"
    },
    "ARTILLERY": {
        "attack": 600, 
        "health": 300, 
        "cost": 60, 
        "max": 20,
        "speed": 1,
        "description": "High attack with medium health"
    },
    "GUARD_TOWER": {
        "attack": 300, 
        "health": 300, 
        "cost": 50, 
        "max": 10,
        "speed": 0,  # Static
        "description": "Static defensive structure"
    }
}

# Budget constraint for each side
MAX_BUDGET = 1500

# Sides
SIDES = ["ENEMY", "HOME"]

# Visual settings
GRID_CELL_SIZE = 20  # Pixels per grid cell
SCREEN_WIDTH = GRID_WIDTH * GRID_CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * GRID_CELL_SIZE

# Colors
COLORS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "BLUE": (0, 0, 255),
    "YELLOW": (255, 255, 0),
    "GREY": (200, 200, 200),
    "DARK_GREY": (100, 100, 100),
    "ENEMY_COLOR": (255, 100, 100),
    "HOME_COLOR": (100, 100, 255),
    "GRID_LINE": (230, 230, 230)
}

# Unit colors
UNIT_COLORS = {
    "SOLDIER": (50, 150, 50),
    "TANK": (100, 100, 100),
    "LANDMINE": (200, 50, 50),
    "FIGHTER_JET": (50, 50, 200),
    "SHIELDED_SOLDIER": (50, 200, 200),
    "ARTILLERY": (200, 100, 50),
    "GUARD_TOWER": (150, 150, 50)
}

# Database settings
DB_PATH = "battle_data.db"

# Model file paths
FORMATION_RECOGNIZER_PATH = "formation_recognizer.pt"
STRATEGY_MODEL_PATH = "strategy_ai_model"

# Training parameters
TRAINING_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0005 