"""
Test script to verify if all required packages are installed correctly.
Run this script to check for import issues.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

print("\nChecking imports:")

# Test numpy
try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except ImportError as e:
    print(f"✗ numpy: {e}")

# Test pygame
try:
    import pygame
    print(f"✓ pygame {pygame.__version__}")
except ImportError as e:
    print(f"✗ pygame: {e}")

# Test torch
try:
    import torch
    print(f"✓ torch {torch.__version__}")
except ImportError as e:
    print(f"✗ torch: {e}")

# Test stable-baselines3
try:
    import stable_baselines3
    print(f"✓ stable_baselines3 {stable_baselines3.__version__}")
    
    from stable_baselines3 import PPO
    print(f"✓ stable_baselines3.PPO")
except ImportError as e:
    print(f"✗ stable_baselines3: {e}")
    print("  To install: pip install stable-baselines3[extra]")

# Test gymnasium (required for stable-baselines3)
try:
    import gymnasium
    print(f"✓ gymnasium {gymnasium.__version__}")
except ImportError as e:
    print(f"✗ gymnasium: {e}")
    print("  To install: pip install gymnasium")

print("\nAll import tests completed.")
print("If you see any ✗ marks above, you need to install those packages.") 