# Battleground Simulator with AI Strategy

A strategic battleground simulation environment with AI model training capabilities.

## Overview

The Battleground Simulator is a comprehensive platform for simulating strategic battles between two sides (Home and Enemy) on a grid-based battlefield. The system combines traditional rule-based AI with modern machine learning approaches, including reinforcement learning, to explore optimal battle strategies.

## Step-by-Step Running Guide

### Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ installed
   ```bash
   python --version
   ```

2. **Required Packages**: Install all dependencies
   ```bash
   pip install -r requirements.txt
   ```
   
   Main dependencies include:
   - PyTorch (ML models)
   - Pygame (visualization)
   - NumPy (numerical operations)
   - Stable-Baselines3 (reinforcement learning, optional)

### Running the Application

1. **Navigate to the project directory**:
   ```bash
   cd battleground_simulator
   ```

2. **Launch the simulator**:
   ```bash
   python src/main.py
   ```

3. **Using the Main Menu**:
   
   When the application starts, you'll see the main menu with these options:
   ```
   1. Run Demo Battle
   2. Run Training Session
   3. Retrain Models
   4. View Battle Statistics
   5. Run Multiple Test Battles
   6. Exit
   ```

4. **Running Your First Battle**:
   - Select option `1` to run a demo battle
   - Choose whether to use reinforcement learning (y/n)
   - Watch the battle unfold in the visualization window
   - Press 'S' at any time to skip to the battle's end result

5. **Training the AI**:
   - Select option `2` to run a training session
   - Enter the number of battles (start with 50-100)
   - The simulator will run battles automatically and collect data
   - Results will be displayed after training completes

6. **Retraining the Models**:
   - After collecting enough data (20+ battles), select option `3`
   - This retrains all the AI models with your collected battle data
   - Choose whether to train the reinforcement learning model (optional)

7. **Analyzing Results**:
   - Select option `4` to view battle statistics
   - You can view plots showing win rates and formation effectiveness
   - This helps understand how the AI is performing

8. **Batch Testing**:
   - Select option `5` to run multiple test battles
   - This is useful for evaluating your trained models

### RL Model Training (Optional)

For advanced reinforcement learning features:

1. **Install RL dependencies**:
   ```bash
   pip install stable-baselines3[extra]
   ```

2. **Run the Application and Train RL Model**:
   - Launch the application as described above
   - Select option `2` to run at least 100 training battles
   - Select option `3` to retrain models
   - When prompted, choose 'y' to train the reinforcement learning model
   - The training will take some time (typically 15-30 minutes)

3. **Test RL Model**:
   - Select option `1` for a demo battle
   - Choose 'y' for using reinforcement learning
   - Observe how the RL model creates formations

### Troubleshooting

- **Missing Sprites**: If sprites don't display, check the sprites directory exists
- **RL Model Errors**: If RL features aren't working, ensure stable-baselines3 is installed
- **Performance Issues**: For slower computers, reduce the number of units in battle

## Application Structure

```
battleground_simulator/
├── src/
│   ├── data/             # Data collection and management
│   ├── models/           # AI models and training
│   ├── simulation/       # Core simulation logic
│   ├── utils/            # Helper functions and constants
│   ├── visualization/    # Visualization components
│   ├── main.py           # Main application entry point
├── sprites/              # Visual assets for units
```

## Core Components

### 1. Battlefield Simulation Engine

The simulation engine manages the battlefield state and executes battle logic:

- **Grid-based battlefield**: Units are placed on a 2D grid
- **Turn-based simulation**: Alternating movement and combat phases
- **Multiple unit types**: Each with unique stats (health, damage, range)
- **Base regions**: Home and Enemy sides with strategic significance

The `BattleSimulator` class handles:
- Formation generation and validation
- Turn execution (movement and combat)
- Outcome determination and health calculation

### 2. Visualization System

The `BattlefieldVisualizer` class provides:
- Real-time battle visualization using Pygame
- Step-by-step replay of battles
- Ability to skip to the battle aftermath by pressing 'S'
- Health bars and status indicators
- Sprite-based unit representation

### 3. AI Strategy Components

The application includes multiple AI approaches:

#### a. Strategy Recommender

The `StrategyRecommender` analyzes enemy formations and recommends counter-formations:
- Uses a trained neural network model to predict success probabilities
- Generates multiple formation recommendations ranked by effectiveness
- Considers unit type interactions and positioning

#### b. Formation Recognizer

Uses pattern recognition to identify common formation strategies from past battles:
- Processes formations into feature vectors
- Classifies formations into strategic patterns
- Helps inform counter-strategy decisions

#### c. Reinforcement Learning

Implements a Proximal Policy Optimization (PPO) agent for advanced strategy learning:
- Learns from battle simulations without explicit programming
- Uses a neural network policy to determine unit placements
- Maximizes battle win rates through trial and error
- Encourages diverse unit type usage through entropy regularization

### 4. Data Collection and Analysis

The `BattleDataCollector` component:
- Records battle outcomes and formation effectiveness
- Tracks win rates and unit performance
- Stores data for model training and analysis
- Generates performance metrics and visualizations

## Application Flow

1. **Initialization**:
   - Load or initialize AI models
   - Set up visualization system
   - Prepare data collection structures

2. **Battle Simulation**:
   - Generate or receive enemy formation
   - Use AI to recommend counter-formations
   - Simulate battle with step-by-step execution
   - Visualize battle progression
   - Determine winner and record results

3. **Training and Learning**:
   - Collect data from multiple battles
   - Train Formation Recognizer model
   - Update Counter-Strategy model
   - Optionally train Reinforcement Learning agent
   - Evaluate model performance

## Key Features

### 1. Battle Visualization

- Step-by-step visualization of battles
- Real-time rendering of unit movements and combat
- Health indicators and battle progress tracking
- Option to skip to battle aftermath by pressing 'S'

### 2. AI Strategy Systems

- Traditional AI recommendations based on pattern matching
- Machine learning models trained on battle data
- Reinforcement learning for adaptive strategy development
- Multiple recommendation options with success probabilities

### 3. Analysis and Improvement

- Win rate tracking and performance analysis
- Unit type effectiveness evaluation
- Formation effectiveness visualization
- Model retraining with new battle data

## Reinforcement Learning Implementation

The RL component uses Stable Baselines3's PPO implementation:

1. **Environment**: Battle simulator wrapped in a Gym-compatible environment
2. **Observation Space**: Enemy formation represented as a grid
3. **Action Space**: Placement decisions for home units
4. **Reward Function**: Based on battle outcome and health difference
5. **Training Process**:
   - Agent explores different formation strategies
   - Learns from battle outcomes through reward signals
   - Gradually improves win rate and unit diversity
   - Uses entropy regularization to encourage exploration

## Usage

The main menu provides the following options:

1. **Run Demo Battle**: Simulate and visualize a single battle
   - Choose between RL agent or traditional AI
   - Watch battle progression (press 'S' to skip to end)

2. **Run Training Session**: Execute multiple battles for training data

3. **Retrain Models**: Update AI models with collected battle data

4. **View Battle Statistics**: Analyze performance metrics 

5. **Run Multiple Test Battles**: Evaluate model performance over many battles

## Technical Design

### Modular Architecture

The simulator follows a modular design pattern:
- Separation of simulation, visualization, and AI components
- Interchangeable model implementations
- Data collection independent of simulation logic

### State Representation

Battlefield state is represented as multi-dimensional arrays:
- Grid positions for spatial relationships
- Unit types and properties encoded as features
- Side information (Home/Enemy) as state attributes

### Algorithm Selection

- **Formation Recognition**: Neural networks for pattern classification
- **Strategy Recommendation**: Supervised learning with battle outcome prediction
- **Unit Placement**: Reinforcement learning with PPO for adaptive strategy

### Performance Considerations

- Optimized battle simulation for rapid training
- Fast visualization modes for efficient testing
- Multi-battle evaluation tools for thorough assessment

## Features

- Grid-based battlefield simulator (40×25 squares)
- Multiple military unit types with different attributes
- AI-powered strategy recommendation system
- Advanced formation analysis and counter-strategy generation
- Battle visualization with Pygame
- Data collection and learning system

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Military Units

| Unit Type | Attack | Health | Cost | Max Units | Notes |
|-----------|--------|--------|------|-----------|-------|
| Soldiers | 200 | 200 | 15 | 50 | Each unit represents 100 soldiers |
| Tanks | 350 | 600 | 50 | 20 | |
| Landmines | 200 | N/A | 10 | 10 | Static defensive units |
| Fighter Jets | 700 | 600 | 100 | 7 | |
| Shielded Soldiers | 100 | 400 | 20 | 50 | Each unit represents 50 soldiers |
| Artillery Vehicles | 600 | 300 | 60 | 20 | |
| Guard Towers | 300 | 300 | 50 | 10 | |

Budget constraint: Maximum cost for each side is 1500 units.

## Project Structure

- `src/` - Source code
  - `models/` - AI models and machine learning components
  - `simulation/` - Battle simulation engine
  - `visualization/` - Visualization components
  - `utils/` - Utility functions
  - `data/` - Data collection and management
  - `strategies/` - Formation strategies and patterns
- `sprites/` - Visual assets for battlefield units 