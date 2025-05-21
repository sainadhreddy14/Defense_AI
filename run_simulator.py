"""
Runs the battleground simulator application.
"""

import os
import sys

# Check for stable-baselines3
try:
    import stable_baselines3
    print(f"Found stable-baselines3 version {stable_baselines3.__version__}")
except ImportError:
    print("Note: stable-baselines3 not found. Reinforcement learning features will be disabled.")
    print("To enable all features, install: pip install stable-baselines3[extra]")

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main function from src.main
from src.main import main

if __name__ == "__main__":
    main() 