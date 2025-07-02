# Training Configuration
TRAINING_CONFIG = {
    'max_episodes': 1000,        # Maximum number of episodes to train
    'update_target_every': 10,  # How often to update target network
    'snake_speed': 50000,           # Game speed
    'save_model': True,          # Whether to save the trained model
    'model_filename': 'snake_dqn_model.pth',  # Filename for saved model
}

# DQN Hyperparameters
DQN_CONFIG = {
    'epsilon_start': 1.0,        # Initial exploration rate
    'epsilon_min': 0.01,         # Minimum exploration rate
    'epsilon_decay': 0.99,      # Exploration decay rate
    'learning_rate': 0.001,      # Learning rate for optimizer
    'gamma': 0.95,               # Discount factor
    'batch_size': 32,            # Training batch size
    'memory_size': 1000000,        # Experience replay buffer size
    'hidden_size': 64            # Number of neurons in hidden layers
}

# Game Configuration
GAME_CONFIG = {
    'window_x': 720,             # Game window width
    'window_y': 480,             # Game window height
    'grid_size': 10              # Size of each grid cell
}