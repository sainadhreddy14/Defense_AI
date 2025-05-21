# Machine Learning Logic in Battleground Simulator

This document provides an in-depth exploration of the machine learning components and algorithms implemented in the Battleground Simulator application.

## ML Architecture Overview

The Battleground Simulator employs multiple machine learning approaches working in concert:

1. **Formation Recognition** - Pattern classification using neural networks
2. **Counter-Strategy Prediction** - Supervised learning for outcome prediction
3. **Reinforcement Learning** - Deep RL for adaptive strategy formation

Each component serves a specific purpose in the AI decision-making pipeline, with information flowing between systems to create a comprehensive strategic intelligence.

## Data Representation

### Formation Encoding

Formations are represented as 3D tensors with dimensions:
- Height (grid rows)
- Width (grid columns)
- Unit type channels

```
formation_tensor = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(UNIT_TYPES)))
```

Each cell contains either 0 (empty) or 1 (unit present), creating a sparse, multi-channel binary grid representation.

### Feature Engineering

Raw formations are processed into feature vectors capturing:

1. **Spatial Features**:
   - Unit density distributions (horizontal/vertical)
   - Distance-from-base metrics
   - Formation center of mass
   - Clustering coefficients

2. **Tactical Features**:
   - Unit type ratios
   - Offensive/defensive unit balance
   - Range coverage maps
   - Fire concentration potential

3. **Strategic Features**:
   - Formation pattern encodings
   - Historical effectiveness against similar formations
   - Coverage of critical battlefield areas

## Formation Recognition System

### Architecture

The Formation Recognizer uses a convolutional neural network (CNN) structure:

```python
class FormationRecognizer(nn.Module):
    def __init__(self, num_patterns=10):
        super(FormationRecognizer, self).__init__()
        # Input: (batch_size, num_unit_types, grid_height, grid_width)
        self.conv1 = nn.Conv2d(len(UNIT_TYPES), 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate flattened size after convolutions and pooling
        flattened_size = 64 * (GRID_HEIGHT // 4) * (GRID_WIDTH // 4)
        
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, num_patterns)
        
    def forward(self, x):
        # x shape: (batch_size, num_unit_types, grid_height, grid_width)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)
```

### Training Methodology

The Formation Recognizer is trained using supervised learning:

1. **Training Data Preparation**:
   - Collection of formations from historical battles
   - Manual or semi-automated labeling of formation patterns
   - Data augmentation through rotation, reflection, and minor unit displacement

2. **Training Process**:
   - Cross-entropy loss function
   - Adam optimizer with learning rate of 0.001
   - Batch size of 32 formations
   - Early stopping based on validation set performance
   - Stratified 5-fold cross-validation for hyperparameter tuning

3. **Performance Metrics**:
   - Classification accuracy
   - Confusion matrix analysis
   - F1 score for each formation pattern class

### Pattern Recognition Logic

The system identifies common tactical formations such as:
- Frontal assault
- Pincer movement
- Defensive turtling
- Concentrated breakthrough
- Encirclement
- Artillery support
- Scout-heavy reconnaissance

Each pattern has tactical strengths and weaknesses that inform the counter-strategy generation.

## Counter-Strategy Prediction System

### Architecture

The Strategy Recommender uses a combination of CNNs and fully connected networks:

```python
class StrategyPredictor(nn.Module):
    def __init__(self):
        super(StrategyPredictor, self).__init__()
        
        # CNN for processing enemy formation
        self.enemy_conv1 = nn.Conv2d(len(UNIT_TYPES), 32, kernel_size=3, padding=1)
        self.enemy_pool1 = nn.MaxPool2d(kernel_size=2)
        self.enemy_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enemy_pool2 = nn.MaxPool2d(kernel_size=2)
        
        # CNN for processing potential counter formation
        self.counter_conv1 = nn.Conv2d(len(UNIT_TYPES), 32, kernel_size=3, padding=1)
        self.counter_pool1 = nn.MaxPool2d(kernel_size=2)
        self.counter_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.counter_pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate flattened size after convolutions and pooling
        cnn_output_size = 64 * (GRID_HEIGHT // 4) * (GRID_WIDTH // 4)
        
        # Fully connected layers for combined processing
        self.fc1 = nn.Linear(cnn_output_size * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Single output: win probability
        
    def forward(self, enemy_formation, counter_formation):
        # Process enemy formation
        e = F.relu(self.enemy_conv1(enemy_formation))
        e = self.enemy_pool1(e)
        e = F.relu(self.enemy_conv2(e))
        e = self.enemy_pool2(e)
        e = e.view(e.size(0), -1)  # Flatten
        
        # Process counter formation
        c = F.relu(self.counter_conv1(counter_formation))
        c = self.counter_pool1(c)
        c = F.relu(self.counter_conv2(c))
        c = self.counter_pool2(c)
        c = c.view(c.size(0), -1)  # Flatten
        
        # Concatenate both formation representations
        combined = torch.cat((e, c), dim=1)
        
        # Process through fully connected layers
        x = F.relu(self.fc1(combined))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for probability output
        
        return x
```

### Training Methodology

The Counter-Strategy model is trained on paired formations with known outcomes:

1. **Training Data Collection**:
   - Pairs of (enemy_formation, home_formation) from historical battles
   - Battle outcome labels: win (1), loss (0), or victory margin as a continuous value
   - Battle health differentials as additional regression targets

2. **Loss Function**:
   - Binary cross-entropy for win/loss classification
   - Mean squared error for health differential prediction
   - Combined loss with weighting parameters

3. **Training Regimen**:
   - Curriculum learning approach: starting with clear victories/defeats, gradually introducing more nuanced battles
   - Regularization through L2 weight decay (0.0001)
   - Batch normalization between layers
   - Learning rate scheduling with cosine annealing

### Formation Generation Logic

The Strategy Recommender generates counter-formations through:

1. **Candidate Generation**:
   - Template-based approach using known effective formations
   - Variations through unit substitution and position adjustment
   - Random exploration with constraints

2. **Candidate Evaluation**:
   - Each candidate formation is scored using the prediction model
   - Top-k formations are selected based on predicted win probability
   - Diversity enforcement through maximum diversity sampling

3. **Budget Constraints**:
   - Each formation must satisfy the maximum cost constraint
   - Unit count limits are enforced by type
   - Valid positioning rules are applied

## Reinforcement Learning System

### Environment Design

The RL component wraps the battle simulator in a Gym-compatible environment:

```python
class BattleEnvironment(gym.Env):
    def __init__(self):
        super(BattleEnvironment, self).__init__()
        
        # Simulator instance
        self.simulator = BattleSimulator()
        
        # Define observation space (enemy formation)
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(GRID_HEIGHT, GRID_WIDTH, len(UNIT_TYPES)),
            dtype=np.float32
        )
        
        # Define action space (home unit placement)
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(GRID_HEIGHT, GRID_WIDTH, len(UNIT_TYPES)),
            dtype=np.float32
        )
        
        # Current state
        self.enemy_formation = None
        
    def reset(self):
        # Generate random enemy formation
        self.enemy_formation = self.simulator.generate_random_formation("ENEMY")
        return np.array(self.enemy_formation, dtype=np.float32)
    
    def step(self, action):
        # Convert continuous action values to binary formation
        home_formation = self._process_action(action)
        
        # Ensure formation is valid
        home_formation = self._make_formation_valid(home_formation)
        
        # Simulate battle
        winner, enemy_health, home_health = self.simulator.simulate_battle(
            self.enemy_formation, home_formation
        )
        
        # Calculate reward
        reward = self._calculate_reward(winner, enemy_health, home_health)
        
        # Battle is done after one step
        done = True
        
        # Additional info
        info = {
            'winner': winner,
            'enemy_health': enemy_health,
            'home_health': home_health,
            'health_diff': home_health - enemy_health
        }
        
        return np.array(self.enemy_formation, dtype=np.float32), reward, done, info
    
    def _process_action(self, action):
        # Convert continuous actions to binary placement (0 or 1)
        # using thresholding and considering unit type constraints
        threshold = 0.5
        binary_action = (action > threshold).astype(np.float32)
        return binary_action
    
    def _make_formation_valid(self, formation):
        # Enforce budget constraints
        # Enforce unit count limits
        # Enforce valid positioning
        # This is a complex process implemented in the simulator
        return formation
    
    def _calculate_reward(self, winner, enemy_health, home_health):
        # Base reward for win/loss
        if winner == "HOME":
            base_reward = 1.0
        elif winner == "ENEMY":
            base_reward = -1.0
        else:  # DRAW
            base_reward = 0.0
        
        # Health differential reward component
        health_diff = home_health - enemy_health
        health_diff_norm = health_diff / 5000.0  # Normalize to [-1, 1] range
        
        # Combined reward with weightings
        reward = 0.7 * base_reward + 0.3 * health_diff_norm
        
        return reward
```

### PPO Implementation

The Proximal Policy Optimization (PPO) algorithm is used for training:

1. **Network Architecture**:
   - Actor-Critic architecture with shared feature extraction
   - Policy network outputs unit placement probabilities
   - Value network estimates expected returns

2. **PPO Specific Parameters**:
   - Clipping parameter: 0.2
   - Value function coefficient: 0.5
   - Entropy coefficient: 0.01 (higher than default to encourage diversity)
   - Learning rate: 3e-4 with linear decay
   - GAE-Lambda: 0.95
   - Discount factor (gamma): 0.99

3. **Training Process**:
   ```python
   def train_strategy_ai(num_iterations=2000):
       # Create environment
       env = BattleEnvironment()
       
       # Define PPO model with increased entropy coefficient
       model = PPO(
           "MlpPolicy",
           env,
           verbose=1,
           tensorboard_log="./ppo_battleground_tensorboard/",
           ent_coef=0.01,  # Higher entropy to encourage exploration
           learning_rate=3e-4,
           n_steps=2048,
           batch_size=64,
           n_epochs=10,
           clip_range=0.2
       )
       
       # Train model with progress bar
       model.learn(total_timesteps=num_iterations, progress_bar=True)
       
       # Save model
       model.save(STRATEGY_MODEL_PATH)
       
       return model
   ```

### Training Challenges and Solutions

1. **Sparse Reward Problem**:
   - Challenge: Binary win/loss rewards provide limited learning signal
   - Solution: Reward shaping with health differentials and turn-based intermediate rewards

2. **Exploration vs. Exploitation**:
   - Challenge: Agent may converge to sub-optimal strategies
   - Solution: Increased entropy coefficient and curriculum learning approach

3. **Overfitting to Specific Enemies**:
   - Challenge: Agent may specialize against common formations but fail against novel ones
   - Solution: Diversified enemy formations during training and adversarial generation

4. **Unit Type Diversity**:
   - Challenge: Agent tends to converge on using only a few unit types
   - Solution: Diversity bonuses in the reward function and entropy regularization

## Integration of ML Components

The three ML systems operate in a coordinated workflow:

1. **Formation Recognition → Strategy Recommendation**:
   - Recognition of enemy formation patterns provides features for the strategy recommender
   - Historical performance against similar formations informs recommendation confidence

2. **Strategy Recommendation → RL Refinement**:
   - Rule-based formations serve as initial guidance for RL policy
   - RL agent can refine and adapt the recommended strategies

3. **RL Feedback → Strategy Database**:
   - Successful RL-discovered formations are added to the strategy database
   - These formations become templates for future strategy recommendations

## Evaluation Metrics

The ML systems are evaluated using multiple metrics:

1. **Win Rate**: Percentage of battles won against random or specific enemy formations
2. **Unit Diversity**: Distribution of unit types used in generated formations
3. **Battle Efficiency**: Health remaining after victory or health differential in defeat
4. **Adaptation Speed**: How quickly the system adapts to new enemy strategies
5. **Formation Stability**: Consistency of recommendations for similar enemy formations

## Technical Implementation Details

### Optimization Techniques

1. **Batch Processing**:
   - Formation evaluation in parallel batches
   - GPU acceleration for neural network inference

2. **Caching Mechanisms**:
   - Memoization of formation evaluations
   - Storage of frequently encountered patterns

3. **Progressive Training**:
   - Models initially trained on simplified battles
   - Gradually increasing complexity and enemy sophistication

### Hyperparameter Optimization

Key hyperparameters were tuned using:
- Grid search for neural network architectures
- Random search for learning rates and regularization parameters
- Manual tuning for RL-specific parameters based on empirical results

Results showed that entropy coefficient and reward shaping had the most significant impact on RL performance.

## Future ML Enhancements

1. **Meta-Learning Approaches**:
   - Learning to adapt to new enemy strategies with minimal battles
   - Fast adaptation through meta-gradients

2. **Multi-Agent Reinforcement Learning**:
   - Training against self-play or evolving opponent populations
   - Emergence of complex strategies through agent competition

3. **Explainable AI Components**:
   - Visualization of decision factors in strategy selection
   - Natural language explanation of tactical recommendations

4. **Transfer Learning**:
   - Pre-training on simplified battle scenarios
   - Fine-tuning for complex, full-scale battles

## Conclusion

The machine learning pipeline in the Battleground Simulator integrates multiple AI paradigms to create a comprehensive strategy recommendation system. The combination of supervised learning for pattern recognition, predictive modeling for counter-strategy generation, and reinforcement learning for adaptive strategy refinement creates a robust and flexible AI opponent that can continually improve through battle experience. 