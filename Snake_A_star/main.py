import pygame
import random
import numpy as np
import heapq
import asyncio

snake_speed = 60


window_x = 720
window_y = 480


black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

game_state = 'TITLE'  # 'TITLE', 'RUNNING', 'GAME_OVER'

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


pygame.init()
pygame.display.set_caption('Snakes')
game_window = pygame.display.set_mode((window_x, window_y))
fps = pygame.time.Clock()






def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)

    score_rect = score_surface.get_rect()

    game_window.blit(score_surface, score_rect)


# --- Button Drawing Function ---
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


# --- Title Screen ---
def title_screen():
    game_window.fill(black)
    title_font = pygame.font.SysFont('times new roman', 60)
    title_surface = title_font.render('Snake Game', True, green)
    title_rect = title_surface.get_rect(center=(window_x / 2, window_y / 4))
    game_window.blit(title_surface, title_rect)
    draw_button(game_window, 'Start Game', window_x / 2 - 100, window_y / 2, 200, 50, blue, green, start_game)
    pygame.display.update()


# --- Game Over Screen ---
def game_over_screen():
    game_window.fill(black)
    font = pygame.font.SysFont('times new roman', 50)
    over_surface = font.render(f'Game Over! Score: {score}', True, red)
    over_rect = over_surface.get_rect(center=(window_x / 2, window_y / 4))
    game_window.blit(over_surface, over_rect)
    draw_button(game_window, 'Restart', window_x / 2 - 100, window_y / 2, 200, 50, blue, green, restart_game)
    pygame.display.update()


# --- Game Start/Restart Logic ---
def start_game():
    global game_state
    game_state = 'RUNNING'


def restart_game():
    global snake_position, snake_body, fruit_position, fruit_spawn, direction, change_to, score, game_state
    snake_position = [100, 50]
    snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]
    fruit_position = [random.randrange(1, (window_x // 10)) * 10,
                      random.randrange(1, (window_y // 10)) * 10]
    fruit_spawn = True
    direction = 'RIGHT'
    change_to = direction
    score = 0
    game_state = 'RUNNING'


# Replace the game_over() function with:
def game_over():
    global game_state
    game_state = 'GAME_OVER'

class Node:
    def __init__(self, row, col, direction, initial_move, g, f):
        self.row = row
        self.col = col
        self.direction = direction
        self.initial_move = initial_move
        self.g = g
        self.f = f

    def __lt__(self, other):
        return self.f < other.f

def reverse_direction(direction: int) -> int:
    return {
        0: 1,  # up -> down
        1: 0,  # down -> up
        2: 3,  # left -> right
        3: 2   # right -> left
    }.get(direction, -1)

def out_of_bounds(board: np.ndarray, row: int, col: int) -> bool:
    rows, cols = board.shape
    if row < 0 or col < 0 or row >= rows or col >= cols:
        return False
    else:
        return True

def available_space(board: np.ndarray, row: int, col: int) -> bool:
    return board[row, col] == 0 or board[row, col] == 2

def is_valid_move(board: np.ndarray, row: int, col: int, visited, direction) -> bool:
    if not out_of_bounds(board, row, col):
        return False
    if not available_space(board, row, col):
        return False
    if visited[row][col][direction]:
        return False
    return True

def estimate_heuristic(current_pos, target_pos) -> int:
    x, y = current_pos
    manhattan_dist = abs(x - target_pos[0]) + abs(y - target_pos[1])
    return int(manhattan_dist)

def short_move(board: np.ndarray, start, target):
    rows, cols = board.shape

    directions = [
        (0, -1),  # up
        (0, 1),   # down
        (-1, 0),  # left
        (1, 0)    # right
    ]

    visited = np.zeros((rows, cols, 4), dtype=bool)
    open_list = []

    for i, (dx, dy) in enumerate(directions):
        new_row = start[1] + dy
        new_col = start[0] + dx
        if (new_col, new_row) == target and is_valid_move(board, new_row, new_col, visited, i):
            return [i, 1]

        visited[start[1], start[0], i] = True

        if is_valid_move(board, new_row, new_col, visited, i):
            g_cost = 1
            h_cost = estimate_heuristic((new_col, new_row), target)
            heapq.heappush(open_list, Node(new_row, new_col, i, i, g_cost, g_cost + h_cost))
            visited[new_row, new_col, i] = True
            visited[start[1], start[0], reverse_direction(i)] = True

    while open_list:
        current = heapq.heappop(open_list)

        for i, (dx, dy) in enumerate(directions):
            new_row = current.row + dy
            new_col = current.col + dx

            if (new_col, new_row) == target:
                return [current.initial_move, current.g]

            if is_valid_move(board, new_row, new_col, visited, i):
                g_cost = current.g + 1
                h_cost = estimate_heuristic((new_col, new_row), target)
                heapq.heappush(open_list, Node(new_row, new_col, i, current.initial_move, g_cost, g_cost + h_cost))
                visited[new_row, new_col, i] = True
                visited[current.row, current.col, reverse_direction(i)] = True

    return [-1, 0]

def print_board(board):
    output = ""
    for row in board:
        for cell in row:
            output += str(cell) + " "
        output += "\n"

    print(output)


def get_move(snake_body, fruit_position, snake_position, direction):
    board = np.zeros((window_y//10, window_x//10), dtype=int)
    fruit_x, fruit_y = fruit_position
    fruit_x = fruit_x//10
    fruit_y = fruit_y//10
    board[fruit_y][fruit_x] = 2
    for i in snake_body:
        board[(i[1]//10)][(i[0]//10)] = 1

    short = short_move(board, (snake_position[0]//10, snake_position[1]//10), (fruit_x, fruit_y))[0]



    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    dir = directions[short]

    return dir



# Main Function
async def main():
    global score, fruit_position, direction, fruit_spawn
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

        change_to = get_move(snake_body, fruit_position, snake_position, direction)
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
        # if fruits and snakes collide then scores will be
        # incremented by 10
        snake_body.insert(0, list(snake_position))
        if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
            score += 10
            fruit_spawn = False
        else:
            snake_body.pop()

        if not fruit_spawn:
            fruit_position = [random.randrange(1, (window_x // 10)) * 10,
                              random.randrange(1, (window_y // 10)) * 10]

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

        # displaying score continuously
        show_score(1, white, 'times new roman', 20)


        pygame.display.update()
        fps.tick(snake_speed)
        await asyncio.sleep(0)


asyncio.run(main())



