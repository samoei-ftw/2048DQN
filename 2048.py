import random
import numpy as np
import pygame
import torch.optim as optim
from collections import deque
from dqn import DQN
from game_play import GamePlay

epsilon = 1.0  # Exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995 
learning_rate = 0.001
batch_size = 64
replay_buffer = deque(maxlen=2000) # stores agent's experiences
def setup_game():
        pygame.init()
        width, height = 400, 400
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('2048 with DQN Agent')
        font = pygame.font.SysFont('arial', 30)

        BACKGROUND_COLOR = (187, 173, 160)
        TILE_COLORS = {
            0: (204, 192, 179), 2: (238, 228, 218), 4: (237, 224, 200), 
            8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
            64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
            512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)
        }
        TEXT_COLOR = (119, 110, 101)
    
def reset_game():
        board = np.zeros((4, 4), dtype=int)
        board = add_random_tile(board)
        board = add_random_tile(board)
        return board

def draw_board(screen, board):
    screen.fill(BACKGROUND_COLOR)
    tile_size = width // 4
    for i in range(4):
        for j in range(4):
            value = board[i][j]
            color = TILE_COLORS.get(value, TILE_COLORS[2048])
            pygame.draw.rect(screen, color, (j * tile_size, i * tile_size, tile_size, tile_size))
            if value > 0:
                text = font.render(str(value), True, TEXT_COLOR)
                text_rect = text.get_rect(center=(j * tile_size + tile_size / 2, i * tile_size + tile_size / 2))
                screen.blit(text, text_rect)
    pygame.display.update()

def select_action(state):
    if random.random() < epsilon:
        return random.randint(0, 3)  # Explore: choose a random action
    else:
        with torch.no_grad():
            return policy_net(state).argmax().item()  # Exploit: choose action with max Q-value

def merge_tiles(row):
        merged_row = []
        skip = False
        for i in range(len(row)):
            if skip:
                skip = False
                continue
            if i + 1 < len(row) and row[i] == row[i + 1]:
                merged_row.append(row[i] * 2)
                skip = True
            else:
                merged_row.append(row[i])
        return np.array(merged_row)

def game_logic(action, current_state, board):
    next_state = np.copy(current_state)

    if action == 0:  # Move up
        next_state, reward = move(next_state)
    elif action == 1:  # Move down
        next_state, reward = move(next_state)
    elif action == 2:  # Move left
        next_state, reward = move(next_state)
    elif action == 3:  # Move right
        next_state, reward = move(next_state)

    # After making a move, add a new tile (2 or 4) at a random empty position
    add_new_tile(next_state)

    # Check if the game is over (no valid moves left)
    done = is_game_over(next_state)

    return next_state, reward, done

def move(board, action):
        if action == 0:  # Left
            for i in range(4):
                row = board[i][board[i] != 0]  # Remove zeros
                row = merge_tiles(row)
                board[i] = np.pad(row, (0, 4 - len(row)), 'constant')
        elif action == 1:  # Right
            for i in range(4):
                row = board[i][board[i] != 0][::-1]  # Remove zeros and reverse
                row = merge_tiles(row)
                board[i] = np.pad(row, (4 - len(row), 0), 'constant')[::-1]
        elif action == 2:  # Up
            for j in range(4):
                col = board[:, j][board[:, j] != 0]  # Remove zeros
                col = merge_tiles(col)
                board[:, j] = np.pad(col, (0, 4 - len(col)), 'constant')
        elif action == 3:  # Down
            for j in range(4):
                col = board[:, j][board[:, j] != 0][::-1]  # Remove zeros and reverse
                col = merge_tiles(col)
                board[:, j] = np.pad(col, (4 - len(col), 0), 'constant')[::-1]
        return board
def step_game(action, current_state, board):
    something = select_action
    # Update the game state based on the action
    next_state, reward, done = game_logic(action, board)

    # Check for the game-over conditions
    if is_game_over(next_state):
        done = True

    return next_state, reward, done
def optimize_model():
    if len(replay_buffer) < batch_size:
        return

    # select random batch of transitions from replay buffer
    batch = random.sample(replay_buffer, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)

    # Convert to tensors
    state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32) # states from sampled transitions
    action_batch = torch.tensor(action_batch, dtype=torch.int64) 
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
    next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32) # next state from resulting action

    current_q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

    next_q_values = target_net(next_state_batch).max(1)[0].detach() # detach: no gradients calculated   
    # apply bellman's equation
    target_q_values = reward_batch + (gamma * next_q_values)

    # Compute the loss (mean squared error)
    loss = nn.MSELoss()(current_q_values, target_q_values)

    optimizer.zero_grad() # clear old gradients from prev step
    loss.backward() # gradient of loss
    optimizer.step() # update network params using gradients

def train_dqn():
    # Training loop
    num_episodes = 1000
    for episode in range(num_episodes):
        state = reset_game()
        done = False
    
    while not done:
        action = select_action(state)
        next_state, reward, done = step_game(action, state)  # Take the action
        replay_buffer.append((state, action, reward, next_state))
        state = next_state
        optimize_model()

    # Update target network periodically
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

def get_current_state():
    return 1

# Main loop
def play_game_with_dqn(policy_net, target_net, gamma, optimizer):
    board = reset_game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        state = np.array(board)  # Convert board to state
        action = select_action(state) #select_action(policy_net, state)
        
        board = move(board, action)
        board = add_random_tile(board)
        
        # new board state
        draw_board(screen, board)
        
        # TODO: Game over condition
        
if __name__ == "__main__":
    train_dqn()
    setup_game()
    policy_net = DQN()
    target_net = DQN() # copy of policy_net but updates less frequently for stability
    target_net.load_state_dict(policy_net.state_dict())  # Initialize target network with same weights
    target_net.eval()
    gamma = 0.99 # discount factor
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    play_game_with_dqn(policy_net, target_net, gamma, optimizer)
