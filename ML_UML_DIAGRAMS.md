# Machine Learning UML Diagrams for Battleground Simulator

This file contains PlantUML code for visualizing the machine learning architecture of the Battleground Simulator.

## Class Diagram

```plantuml
@startuml "ML Components Class Diagram"

package "Core Components" {
  class BattleSimulator {
    + simulate_battle()
    + simulate_battle_with_history()
    + generate_random_formation()
  }
  
  class BattlefieldVisualizer {
    + render_battlefield()
    + render_battle_replay()
    + render_battle_aftermath()
  }
  
  class BattleDataCollector {
    + record_battle()
    + get_battle_data()
    + get_battle_count()
  }
}

package "Machine Learning" {
  class FormationRecognizer {
    - conv1: Conv2d
    - pool1: MaxPool2d
    - conv2: Conv2d
    - pool2: MaxPool2d
    - fc1: Linear
    - fc2: Linear
    + forward()
    + classify_formation()
    + extract_features()
  }
  
  class StrategyRecommender {
    - strategy_predictor: StrategyPredictor
    - formation_templates: List
    + recommend_formations()
    + evaluate_formation()
    + generate_candidates()
    - _make_formation_valid()
  }
  
  class StrategyPredictor {
    - enemy_conv1/2: Conv2d
    - enemy_pool1/2: MaxPool2d
    - counter_conv1/2: Conv2d
    - counter_pool1/2: MaxPool2d
    - fc1/2/3: Linear
    + forward()
    + predict_success_probability()
  }
  
  class BattleEnvironment {
    - simulator: BattleSimulator
    - observation_space: Box
    - action_space: Box
    - enemy_formation: Array
    + reset()
    + step()
    - _process_action()
    - _make_formation_valid()
    - _calculate_reward()
  }
  
  class PPOAgent {
    - policy_network: MlpPolicy
    - value_network: MlpNetwork
    + predict()
    + learn()
    + save()
    + load()
  }
}

package "Training Functions" {
  class train_formation_recognizer {
    + preprocessing()
    + train_model()
    + evaluate_model()
  }
  
  class train_counter_strategy_model {
    + data_preparation()
    + train_model()
    + save_model()
  }
  
  class train_strategy_ai {
    + create_environment()
    + configure_ppo()
    + train_ppo()
    + save_model()
  }
}

BattleSimulator -- BattleDataCollector : provides data to >
BattleDataCollector -- FormationRecognizer : trains <
BattleDataCollector -- StrategyPredictor : trains <

StrategyRecommender -- StrategyPredictor : uses >
StrategyRecommender -- FormationRecognizer : uses >

BattleEnvironment -- BattleSimulator : wraps >
PPOAgent -- BattleEnvironment : interacts with >

train_formation_recognizer -- FormationRecognizer : creates >
train_counter_strategy_model -- StrategyPredictor : creates >
train_strategy_ai -- PPOAgent : creates >
train_strategy_ai -- BattleEnvironment : creates >

@enduml
```

## Sequence Diagram: Formation Recognition and Strategy Recommendation

```plantuml
@startuml "Formation Recognition Workflow"

actor User
participant BattlegroundSimulator
participant BattleSimulator
participant FormationRecognizer
participant StrategyRecommender
participant StrategyPredictor

User -> BattlegroundSimulator : run_demo_battle()
activate BattlegroundSimulator

BattlegroundSimulator -> BattleSimulator : generate_random_formation("ENEMY")
activate BattleSimulator
BattleSimulator --> BattlegroundSimulator : enemy_formation
deactivate BattleSimulator

alt Use AI strategy
  BattlegroundSimulator -> FormationRecognizer : classify_formation(enemy_formation)
  activate FormationRecognizer
  FormationRecognizer --> BattlegroundSimulator : formation_pattern
  deactivate FormationRecognizer
  
  BattlegroundSimulator -> StrategyRecommender : recommend_formations(enemy_formation)
  activate StrategyRecommender
  
  loop for multiple candidate formations
    StrategyRecommender -> StrategyRecommender : generate_candidates()
    
    loop for each candidate
      StrategyRecommender -> StrategyPredictor : predict_success_probability(enemy_formation, candidate)
      activate StrategyPredictor
      StrategyPredictor --> StrategyRecommender : success_probability
      deactivate StrategyPredictor
    end
    
    StrategyRecommender -> StrategyRecommender : select_top_formations()
  end
  
  StrategyRecommender --> BattlegroundSimulator : recommended_formations
  deactivate StrategyRecommender
  
  BattlegroundSimulator -> BattlegroundSimulator : home_formation = recommended_formations[0]
end

BattlegroundSimulator -> BattleSimulator : simulate_battle_with_history(enemy_formation, home_formation)
activate BattleSimulator
BattleSimulator --> BattlegroundSimulator : battle_history
deactivate BattleSimulator

BattlegroundSimulator --> User : battle_visualization
deactivate BattlegroundSimulator

@enduml
```

## Sequence Diagram: Reinforcement Learning Workflow

```plantuml
@startuml "Reinforcement Learning Workflow"

actor User
participant BattlegroundSimulator
participant BattleEnvironment
participant PPOAgent
participant BattleSimulator

User -> BattlegroundSimulator : run_demo_battle(use_rl=True)
activate BattlegroundSimulator

BattlegroundSimulator -> BattleSimulator : generate_random_formation("ENEMY")
activate BattleSimulator
BattleSimulator --> BattlegroundSimulator : enemy_formation
deactivate BattleSimulator

BattlegroundSimulator -> PPOAgent : predict(enemy_formation)
activate PPOAgent

PPOAgent -> PPOAgent : process observation
note right: Convert formation to \ntensor representation

PPOAgent -> PPOAgent : forward pass through policy network
note right: Actor network produces\nunit placement probabilities

PPOAgent --> BattlegroundSimulator : action
deactivate PPOAgent

BattlegroundSimulator -> BattlegroundSimulator : convert action to home_formation
note right: Threshold probabilities and\napply formation constraints

BattlegroundSimulator -> BattleSimulator : simulate_battle_with_history(enemy_formation, home_formation)
activate BattleSimulator
BattleSimulator --> BattlegroundSimulator : battle_history
deactivate BattleSimulator

BattlegroundSimulator --> User : battle_visualization
deactivate BattlegroundSimulator

@enduml
```

## Training Process Diagram

```plantuml
@startuml "ML Training Process"

start

:Initialize Data Collector;

repeat
  :Run training battles;
  :Record battle outcomes and formations;
repeat while (Enough training data?) is (no)
->yes;

fork
  :Train Formation Recognizer;
  :Extract formation patterns;
  :Train CNN model;
  :Evaluate pattern classification accuracy;
fork again
  :Train Counter-Strategy Predictor;
  :Prepare paired formation data;
  :Train CNN with paired formations;
  :Evaluate win prediction accuracy;
fork again
  :Train Reinforcement Learning Agent;
  :Create battle environment;
  :Configure PPO algorithm;
  :Train with exploration;
  :Evaluate against test formations;
end fork

:Save trained models;

:Evaluate integrated system;
:Record performance metrics;

stop

@enduml
```

## Data Flow Diagram

```plantuml
@startuml "ML Data Flow"

agent User
database "Battle\nHistory" as History
database "Formation\nTemplates" as Templates

frame "ML Pipeline" {
  component "Formation\nRecognizer" as FR
  component "Strategy\nRecommender" as SR
  component "Reinforcement\nLearning" as RL
  component "Battle\nSimulator" as BS
}

User --> BS : Enemy formation
BS --> History : Store battle data
History --> FR : Training data
FR --> SR : Formation patterns
Templates --> SR : Formation templates
SR --> BS : Recommended formations
RL --> BS : RL-generated formations
BS --> User : Battle outcome

@enduml
```

## Component Interaction Diagram

```plantuml
@startuml "ML Component Interactions"

package "Machine Learning Pipeline" {
  [Formation Recognizer] as FR
  [Strategy Recommender] as SR
  [Reinforcement Learning Agent] as RL
  [Battle Simulator] as BS
  [Data Collector] as DC
}

interface "Enemy Formation" as EF
interface "Counter Formation" as CF
interface "Battle History" as BH
interface "Training Data" as TD
interface "Win Prediction" as WP
interface "Battle Outcome" as BO

EF --> FR
EF --> SR
EF --> RL

FR --> SR : Pattern features
SR --> CF : Recommended formations
RL --> CF : RL-based formations

CF --> BS
EF --> BS

BS --> BH : Generates
BH --> DC : Stores
DC --> TD : Provides
TD --> FR : Trains
TD --> SR : Trains
BS --> BO : Determines

SR --> WP : Predicts
WP -.-> BO : Compared with

@enduml
```

## ML Decision Process

```plantuml
@startuml "ML Decision Process"

state "Enemy Analysis" as EA {
  state "Formation Recognition" as FR
  state "Historical Effectiveness Analysis" as HEA
}

state "Strategy Generation" as SG {
  state "Template-Based Generation" as TBG
  state "Rule-Based Adaptation" as RBA
  state "RL Policy Generation" as RLPG
}

state "Candidate Evaluation" as CE {
  state "Success Probability Prediction" as SPP
  state "Diversity Analysis" as DA
  state "Budget Validation" as BV
}

state "Strategy Selection" as SS {
  state "Top-K Selection" as TKS
  state "Diversity Enforcement" as DE
}

[*] --> EA
EA --> SG
SG --> CE
CE --> SS
SS --> [*]

@enduml
```

## Reinforcement Learning Architecture

```plantuml
@startuml "Reinforcement Learning Architecture"

package "Battle Environment" {
  [BattleSimulator] as BS
  [RewardCalculator] as RC
  [ActionProcessor] as AP
}

package "PPO Architecture" {
  [PolicyNetwork] as PN
  [ValueNetwork] as VN
  [ExperienceBuffer] as EB
  [EntropyRegularization] as ER
}

package "Training Components" {
  [LossCalculator] as LC
  [Optimizer] as OPT
  [LearningRateScheduler] as LRS
}

BS --> RC : Battle outcome
RC --> EB : Reward signal
EB --> PN : Training data
EB --> VN : Training data
PN --> AP : Action probabilities
AP --> BS : Valid formation
ER --> PN : Encourages exploration
LC --> OPT : Loss values
OPT --> PN : Updates weights
OPT --> VN : Updates weights
LRS --> OPT : Adjusts learning rate

@enduml
```

## Model Interface Diagram

```plantuml
@startuml "ML Model Interfaces"

interface FormationRecognizer {
  + classify_formation(formation: ndarray) -> PatternType
  + extract_features(formation: ndarray) -> ndarray
}

interface StrategyRecommender {
  + recommend_formations(enemy_formation: ndarray, num_recommendations: int) -> List[Formation]
  + evaluate_formation(enemy_formation: ndarray, counter_formation: ndarray) -> float
}

interface RLAgent {
  + predict(observation: ndarray) -> Tuple[ndarray, Dict]
  + learn(total_timesteps: int) -> Self
}

class BattlegroundSimulator {
  + run_demo_battle(enemy_formation: ndarray, use_rl: bool) -> Tuple[str, float, float]
  + run_training_session(num_battles: int) -> float
  + retrain_models() -> None
}

FormationRecognizer <-- BattlegroundSimulator : uses
StrategyRecommender <-- BattlegroundSimulator : uses
RLAgent <-- BattlegroundSimulator : uses

@enduml
```

These UML diagrams provide a comprehensive visualization of the machine learning components and their interactions in the Battleground Simulator. 