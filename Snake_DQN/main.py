import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio

# Game constants
snake_speed = 60
window_x = 720
window_y = 480

# Colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

# Game state
game_state = 'TITLE'  # 'TITLE', 'RUNNING', 'GAME_OVER'

# Game variables
fruit_position = [random.randrange(1, (window_x // 10)) * 10,
                  random.randrange(1, (window_y // 10)) * 10]
fruit_spawn = True
snake_position = [100, 50]
snake_body = [[100, 50],
              [90, 50],
              [80, 50],
              [70, 50]]
direction = 'RIGHT'
change_to = direction
score = 0

# Pygame initialization
pygame.init()
pygame.display.set_caption('Snake DQN Player')
game_window = pygame.display.set_mode((window_x, window_y))
fps = pygame.time.Clock()


# DQN Model (same architecture as training)
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Load trained model
def load_trained_model(model_path='snake_dqn_model.pth'):
    """Load the trained DQN model"""
    try:
        # Initialize model with same architecture as training
        model = DQN(12, 64, 4)  # 12 input features, 64 hidden, 4 actions
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set to evaluation mode
        print(f"Successfully loaded model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Model file {model_path} not found!")
        print("Please train the model first using main.py")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# State representation (same as training)
def get_state(snake_position, snake_body, fruit_position, direction):
    """Create state representation for the AI"""
    state = []

    # Snake head position (normalized)
    state.extend([snake_position[0] / window_x, snake_position[1] / window_y])

    # Fruit position (normalized)
    state.extend([fruit_position[0] / window_x, fruit_position[1] / window_y])

    # Direction (one-hot encoded)
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    direction_one_hot = [1 if direction == d else 0 for d in directions]
    state.extend(direction_one_hot)

    # Danger detection (distance to walls and body in each direction)
    head_x, head_y = snake_position[0] // 10, snake_position[1] // 10

    # Check danger in each direction
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # UP, DOWN, LEFT, RIGHT
        danger = 0
        new_x, new_y = head_x + dx, head_y + dy

        # Check wall collision
        if new_x < 0 or new_x >= window_x // 10 or new_y < 0 or new_y >= window_y // 10:
            danger = 1
        else:
            # Check body collision
            for segment in snake_body[1:]:
                if new_x == segment[0] // 10 and new_y == segment[1] // 10:
                    danger = 1
                    break
        state.append(danger)

    return np.array(state, dtype=np.float32)


# Get action from trained model
def get_dqn_move(model, snake_body, fruit_position, snake_position, direction):
    """Get the best action from the trained DQN model"""
    if model is None:
        # Fallback to random if model not loaded
        return random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])

    # Get current state
    state = get_state(snake_position, snake_body, fruit_position, direction)

    # Convert to tensor and get Q-values
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_tensor)
        action = q_values.argmax().item()

    # Convert action index to direction
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    return directions[action]


def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    game_window.blit(score_surface, score_rect)


# Button Drawing Function
def draw_button(surface, text, x, y, width, height, inactive_color, active_color, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x + width > mouse[0] > x and y + height > mouse[1] > y:
        pygame.draw.rect(surface, active_color, (x, y, width, height))
        if click[0] == 1 and action is not None:
            action()
    else:
        pygame.draw.rect(surface, inactive_color, (x, y, width, height))

    font = pygame.font.SysFont('times new roman', 35)
    text_surface = font.render(text, True, white)
    text_rect = text_surface.get_rect(center=(x + width / 2, y + height / 2))
    surface.blit(text_surface, text_rect)


# Title Screen
def title_screen():
    game_window.fill(black)
    title_font = pygame.font.SysFont('times new roman', 60)
    title_surface = title_font.render('Snake DQN Player', True, green)
    title_rect = title_surface.get_rect(center=(window_x / 2, window_y / 4))
    game_window.blit(title_surface, title_rect)

    # Show model status
    status_font = pygame.font.SysFont('times new roman', 24)
    if dqn_model is not None:
        status_text = "DQN Model: Loaded ✓"
        status_color = green
    else:
        status_text = "DQN Model: Not Found ✗"
        status_color = red

    status_surface = status_font.render(status_text, True, status_color)
    status_rect = status_surface.get_rect(center=(window_x / 2, window_y / 2 - 30))
    game_window.blit(status_surface, status_rect)

    draw_button(game_window, 'Start Game', window_x / 2 - 100, window_y / 2 + 20, 200, 50, blue, green, start_game)
    pygame.display.update()


# Game Over Screen
def game_over_screen():
    game_window.fill(black)
    font = pygame.font.SysFont('times new roman', 50)
    over_surface = font.render(f'Game Over! Score: {score}', True, red)
    over_rect = over_surface.get_rect(center=(window_x / 2, window_y / 4))
    game_window.blit(over_surface, over_rect)
    draw_button(game_window, 'Restart', window_x / 2 - 100, window_y / 2, 200, 50, blue, green, restart_game)
    pygame.display.update()


# Game Start/Restart Logic
def start_game():
    global game_state
    game_state = 'RUNNING'


def restart_game():
    global snake_position, snake_body, fruit_position, fruit_spawn, direction, change_to, score, game_state
    snake_position = [100, 50]
    snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]
    fruit_position = get_valid_fruit_position(snake_body)
    fruit_spawn = True
    direction = 'RIGHT'
    change_to = direction
    score = 0
    game_state = 'RUNNING'


def game_over():
    global game_state
    game_state = 'GAME_OVER'


def get_valid_fruit_position(snake_body):
    """Generate a fruit position that doesn't overlap with the snake body"""
    while True:
        fruit_x = random.randrange(1, (window_x // 10)) * 10
        fruit_y = random.randrange(1, (window_y // 10)) * 10

        # Check if the position overlaps with any part of the snake body
        valid_position = True
        for segment in snake_body:
            if fruit_x == segment[0] and fruit_y == segment[1]:
                valid_position = False
                break

        if valid_position:
            return [fruit_x, fruit_y]


# Main Function
async def main():
    global score, fruit_position, direction, fruit_spawn

    # Load the trained model
    global dqn_model
    dqn_model = load_trained_model()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if game_state == 'TITLE':
            title_screen()
            await asyncio.sleep(0)
            continue
        elif game_state == 'GAME_OVER':
            game_over_screen()
            await asyncio.sleep(0)
            continue

        # Get move from DQN model (replaces A* get_move function)
        change_to = get_dqn_move(dqn_model, snake_body, fruit_position, snake_position, direction)

        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        # Moving the snake
        if direction == 'UP':
            snake_position[1] -= 10
        if direction == 'DOWN':
            snake_position[1] += 10
        if direction == 'LEFT':
            snake_position[0] -= 10
        if direction == 'RIGHT':
            snake_position[0] += 10

        # Snake body growing mechanism
        snake_body.insert(0, list(snake_position))
        if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
            score += 10
            fruit_spawn = False
        else:
            snake_body.pop()

        if not fruit_spawn:
            fruit_position = get_valid_fruit_position(snake_body)

        fruit_spawn = True
        game_window.fill(black)

        for pos in snake_body:
            pygame.draw.rect(game_window, green, pygame.Rect(
                pos[0], pos[1], 10, 10))

        pygame.draw.rect(game_window, red, pygame.Rect(
            fruit_position[0], fruit_position[1], 10, 10))

        # Game Over conditions
        if snake_position[0] < 0 or snake_position[0] > window_x - 10:
            game_over()
        if snake_position[1] < 0 or snake_position[1] > window_y - 10:
            game_over()

        # Touching the snake body
        for block in snake_body[1:]:
            if snake_position[0] == block[0] and snake_position[1] == block[1]:
                game_over()

        # Display score and model info
        show_score(1, white, 'times new roman', 20)

        # Show model status during gameplay
        font = pygame.font.SysFont('times new roman', 16)
        if dqn_model is not None:
            model_text = font.render('DQN AI Playing', True, green)
        else:
            model_text = font.render('Random AI (No Model)', True, red)
        game_window.blit(model_text, (10, 30))

        pygame.display.update()
        fps.tick(snake_speed)
        await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())