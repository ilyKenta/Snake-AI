import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import asyncio
from config import TRAINING_CONFIG, DQN_CONFIG, GAME_CONFIG

snake_speed = TRAINING_CONFIG['snake_speed']

window_x = GAME_CONFIG['window_x']
window_y = GAME_CONFIG['window_y']

black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)
yellow = pygame.Color(255, 255, 0)

game_state = 'TITLE'  # 'TITLE', 'RUNNING', 'GAME_OVER'

snake_position = [100, 50]
snake_body = [[100, 50],
              [90, 50],
              [80, 50],
              [70, 50]]
fruit_position = [random.randrange(1, (window_x // 10)) * 10,
                  random.randrange(1, (window_y // 10)) * 10]
fruit_spawn = True
direction = 'RIGHT'
change_to = direction

score = 0

pygame.init()
pygame.display.set_caption('Snakes')
game_window = pygame.display.set_mode((window_x, window_y))
fps = pygame.time.Clock()


def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)

    score_rect = score_surface.get_rect()

    game_window.blit(score_surface, score_rect)



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



def title_screen():
    game_window.fill(black)
    title_font = pygame.font.SysFont('times new roman', 60)
    title_surface = title_font.render('Snake Game', True, green)
    title_rect = title_surface.get_rect(center=(window_x / 2, window_y / 4))
    game_window.blit(title_surface, title_rect)
    draw_button(game_window, 'Start Game', window_x / 2 - 100, window_y / 2, 200, 50, blue, green, start_game)
    pygame.display.update()



def game_over_screen():
    game_window.fill(black)
    font = pygame.font.SysFont('times new roman', 50)
    over_surface = font.render(f'Game Over! Score: {score}', True, red)
    over_rect = over_surface.get_rect(center=(window_x / 2, window_y / 4))
    game_window.blit(over_surface, over_rect)
    draw_button(game_window, 'Restart', window_x / 2 - 100, window_y / 2, 200, 50, blue, green, restart_game)
    pygame.display.update()



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



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(state),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.BoolTensor(done))

    def __len__(self):
        return len(self.buffer)



class SnakeEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake_position = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]
        self.fruit_position = self.get_valid_fruit_position()
        self.direction = 'RIGHT'
        self.score = 0
        self.game_over = False
        return self.get_state()

    def get_valid_fruit_position(self):
        while True:
            fruit_x = random.randrange(1, (self.width // 10)) * 10
            fruit_y = random.randrange(1, (self.height // 10)) * 10

            valid_position = True
            for segment in self.snake_body:
                if fruit_x == segment[0] and fruit_y == segment[1]:
                    valid_position = False
                    break

            if valid_position:
                return [fruit_x, fruit_y]

    def get_state(self):

        state = []


        state.extend([self.snake_position[0] / self.width, self.snake_position[1] / self.height])


        state.extend([self.fruit_position[0] / self.width, self.fruit_position[1] / self.height])


        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        direction_one_hot = [1 if self.direction == d else 0 for d in directions]
        state.extend(direction_one_hot)


        head_x, head_y = self.snake_position[0] // 10, self.snake_position[1] // 10


        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # UP, DOWN, LEFT, RIGHT
            danger = 0
            new_x, new_y = head_x + dx, head_y + dy

            if new_x < 0 or new_x >= self.width // 10 or new_y < 0 or new_y >= self.height // 10:
                danger = 1
            else:
                for segment in self.snake_body[1:]:
                    if new_x == segment[0] // 10 and new_y == segment[1] // 10:
                        danger = 1
                        break
            state.append(danger)

        return np.array(state, dtype=np.float32)

    def step(self, action):
        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        new_direction = directions[action]


        old_distance = abs(self.snake_position[0] - self.fruit_position[0]) + abs(
            self.snake_position[1] - self.fruit_position[1])


        if (new_direction == 'UP' and self.direction != 'DOWN') or \
                (new_direction == 'DOWN' and self.direction != 'UP') or \
                (new_direction == 'LEFT' and self.direction != 'RIGHT') or \
                (new_direction == 'RIGHT' and self.direction != 'LEFT'):
            self.direction = new_direction


        if self.direction == 'UP':
            self.snake_position[1] -= 10
        elif self.direction == 'DOWN':
            self.snake_position[1] += 10
        elif self.direction == 'LEFT':
            self.snake_position[0] -= 10
        elif self.direction == 'RIGHT':
            self.snake_position[0] += 10


        if (self.snake_position[0] < 0 or self.snake_position[0] >= self.width or
                self.snake_position[1] < 0 or self.snake_position[1] >= self.height):
            self.game_over = True
            return self.get_state(), -200, True


        for segment in self.snake_body[1:]:
            if self.snake_position[0] == segment[0] and self.snake_position[1] == segment[1]:
                self.game_over = True
                return self.get_state(), -200, True


        self.snake_body.insert(0, list(self.snake_position))


        reward = 0
        if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
            self.score += 10
            reward = 50
            self.fruit_position = self.get_valid_fruit_position()
        else:
            self.snake_body.pop()


            new_distance = abs(self.snake_position[0] - self.fruit_position[0]) + abs(
                self.snake_position[1] - self.fruit_position[1])


            if new_distance < old_distance:
                reward = 2  # Reward for moving closer to fruit
            elif new_distance > old_distance:
                reward = -1  # Penalty for moving away from fruit
            else:
                reward = -1  # No change in distance

            # Additional small penalty for each move to encourage efficiency
            reward -= 0.05

        return self.get_state(), reward, False



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.epsilon = DQN_CONFIG['epsilon_start']  # Exploration rate
        self.epsilon_min = DQN_CONFIG['epsilon_min']
        self.epsilon_decay = DQN_CONFIG['epsilon_decay']
        self.learning_rate = DQN_CONFIG['learning_rate']
        self.gamma = DQN_CONFIG['gamma']  # Discount factor
        self.batch_size = DQN_CONFIG['batch_size']
        self.memory = ReplayBuffer(DQN_CONFIG['memory_size'])

        self.q_network = DQN(state_size, DQN_CONFIG['hidden_size'], action_size).to(self.device)
        self.target_network = DQN(state_size, DQN_CONFIG['hidden_size'], action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load_model(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.q_network.state_dict())


def print_board(board):
    output = ""
    for row in board:
        for cell in row:
            output += str(cell) + " "
        output += "\n"

    print(output)


def get_valid_fruit_position(snake_body):
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



env = SnakeEnvironment(window_x, window_y)
state_size = 12
action_size = 4
agent = DQNAgent(state_size, action_size)


max_episodes = TRAINING_CONFIG['max_episodes']  # Maximum number of episodes to train
update_target_every = TRAINING_CONFIG['update_target_every']
training_complete = False



async def main():
    global game_state, snake_position, snake_body, fruit_position, direction, score, training_complete

    episode = 0
    state = env.reset()

    while not training_complete:
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

        state = env.get_state()
        action = agent.act(state)

        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        change_to = directions[action]

        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)

        agent.replay()

        state = next_state

        snake_position = env.snake_position
        snake_body = env.snake_body
        fruit_position = env.fruit_position
        direction = env.direction
        score = env.score
        game_window.fill(black)

        for pos in snake_body:
            pygame.draw.rect(game_window, green, pygame.Rect(
                pos[0], pos[1], 10, 10))

        pygame.draw.rect(game_window, red, pygame.Rect(
            fruit_position[0], fruit_position[1], 10, 10))


        if env.game_over:
            game_over()


        if game_state == 'GAME_OVER':
            episode += 1
            if episode % update_target_every == 0:
                agent.update_target_network()

            if episode >= max_episodes:
                training_complete = True
                print(f"Training completed after {episode} episodes!")
                print(f"Final epsilon: {agent.epsilon:.3f}")
                # Save the trained model
                if TRAINING_CONFIG['save_model']:
                    agent.save_model(TRAINING_CONFIG['model_filename'])
                    print(f"Model saved as '{TRAINING_CONFIG['model_filename']}'")
                break

            state = env.reset()
            game_state = 'RUNNING'


        show_score(1, white, 'times new roman', 20)

        font = pygame.font.SysFont('times new roman', 16)
        epsilon_text = font.render(f'Epsilon: {agent.epsilon:.3f}', True, white)
        episode_text = font.render(f'Episode: {episode}/{max_episodes}', True, white)


        distance = abs(snake_position[0] - fruit_position[0]) + abs(snake_position[1] - fruit_position[1])
        distance_text = font.render(f'Distance to fruit: {distance}', True, white)


        progress = (episode / max_episodes) * 100
        progress_text = font.render(f'Training Progress: {progress:.1f}%', True, white)

        game_window.blit(epsilon_text, (10, 30))
        game_window.blit(episode_text, (10, 50))
        game_window.blit(distance_text, (10, 70))
        game_window.blit(progress_text, (10, 90))

        pygame.display.update()
        fps.tick(snake_speed)
        await asyncio.sleep(0)

    if training_complete:
        await show_training_complete_screen()


async def show_training_complete_screen():
    """Display training completion screen"""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    await play_with_trained_model()
                    return
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

        game_window.fill(black)


        font_large = pygame.font.SysFont('times new roman', 40)
        font_small = pygame.font.SysFont('times new roman', 24)

        title = font_large.render('Training Complete!', True, green)
        subtitle = font_small.render(f'Completed {max_episodes} episodes', True, white)
        instruction1 = font_small.render('Press SPACE to play with trained model', True, white)
        instruction2 = font_small.render('Press ESC to exit', True, white)

        game_window.blit(title, (window_x // 2 - title.get_width() // 2, window_y // 3))
        game_window.blit(subtitle, (window_x // 2 - subtitle.get_width() // 2, window_y // 2))
        game_window.blit(instruction1, (window_x // 2 - instruction1.get_width() // 2, window_y // 2 + 50))
        game_window.blit(instruction2, (window_x // 2 - instruction2.get_width() // 2, window_y // 2 + 80))

        pygame.display.update()
        await asyncio.sleep(0)


async def play_with_trained_model():
    """Play the game using the trained model with minimal exploration"""
    global game_state, snake_position, snake_body, fruit_position, direction, score


    agent.epsilon = 0.01


    state = env.reset()
    game_state = 'RUNNING'

    print("Playing with trained model (epsilon = 0.01)")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return

        if game_state == 'GAME_OVER':
            state = env.reset()
            game_state = 'RUNNING'
            continue


        action = agent.act(state)


        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        change_to = directions[action]

        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'


        next_state, reward, done = env.step(action)
        state = next_state

        snake_position = env.snake_position
        snake_body = env.snake_body
        fruit_position = env.fruit_position
        direction = env.direction
        score = env.score


        if env.game_over:
            game_over()


        game_window.fill(black)

        for pos in snake_body:
            pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

        pygame.draw.rect(game_window, red, pygame.Rect(fruit_position[0], fruit_position[1], 10, 10))


        show_score(1, white, 'times new roman', 20)

        font = pygame.font.SysFont('times new roman', 16)
        mode_text = font.render('TRAINED MODEL MODE - Press ESC to return', True, yellow)
        game_window.blit(mode_text, (10, 30))

        pygame.display.update()
        fps.tick(snake_speed)
        await asyncio.sleep(0)


asyncio.run(main())



