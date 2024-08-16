"""
Script that reads data from DATA PATH and trains a behavior cloning model on it.
Also evaluate the trained model on the environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from pathlib import Path

import gym
import particle_envs

DATA_PATH = Path("./data/straight_fixed_goal_1000_demos.pkl")
SAVE_PATH = Path("./run")
SAVE_PATH.mkdir(exist_ok=True)

class BCModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(BCModel, self).__init__()
        ## number of hidden channels
        hidden_size = 64

        self.hidden = nn.Linear(input_size, hidden_size, bias=True, dtype=torch.float32)
        self.output = nn.Linear(hidden_size, output_size, bias=True, dtype=torch.float32)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        return x

def train_bc_model(env, data_path, model_path):

    # Load data
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    X = data["states"]
    Y = data["actions"]
    goals = data["goals"]

    # Concat states and actions
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    # Convert data to tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # Normalize data
    xmax, xmin = X.max(dim=0)[0], X.min(dim=0)[0]
    ymax, ymin = Y.max(dim=0)[0], Y.min(dim=0)[0]
    X = (X - xmin) / (xmax - xmin)
    Y = (Y - ymin) / (ymax - ymin)

    # Dataloader
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # Create model
    model = BCModel(X.shape[1], Y.shape[1])

    # Define loss and optimizer (aka, loss function!)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train model
    for epoch in range(1000):
        for x, y in dataloader:
          # Zero the gradients
          optimizer.zero_grad()

          # Pass input data into model
          y_hat = model(x)

          # Calculate the loss and gradients
          loss = criterion(y_hat, y)
          loss.backward()

          # Adjust weights
          optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        if epoch % 100 == 0:
            eval(env, goals, xmax, xmin, ymax, ymin, model)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'xmax': xmax,
        'xmin': xmin,
        'ymax': ymax,
        'ymin': ymin
    }, model_path)

    return goals

def eval(env, goals, xmax, xmin, ymax, ymin, model):
    model.eval()
    # Evaluate model
    with torch.no_grad():
        total_reward = 0
        video_frames = []
        for _ in range(10):
            # sample goal
            goal = goals[np.random.randint(len(goals))][0]

            # reset env
            obs = env.reset(reset_goal=True, goal_state=goal)

            done = False
            frames = [env.render(mode='rgb_array', width=256, height=256)]
            while not done:
                obs = torch.tensor(obs, dtype=torch.float32)
                obs = (obs - xmin) / (xmax - xmin)
                action = model(obs[None]).detach()[0]
                action = action * (ymax - ymin) + ymin
                action = action.numpy()
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                frames.append(env.render(mode='rgb_array', width=256, height=256))
            video_frames.extend(frames)
        print(f"Average reward: {total_reward / 10}")

    # Save video
    import imageio as iio
    video_frames = np.array(video_frames).astype(np.uint8)[..., 0]
    iio.mimwrite(SAVE_PATH / "bc_video.mp4", video_frames, fps=10)

    model.train()


if __name__ == "__main__":
    # Initialize environment
    env = gym.make('particle-v0', height=200, width=200, step_size=10,
                    reward_type='dense', reward_scale=None, block=None)

    goals = train_bc_model(env, DATA_PATH, SAVE_PATH / "bc_model.pth")

    # # Load model
    # model = BCModel(2, 2)
    # load = torch.load("bc_model.pth")
    # model.load_state_dict(load['model_state_dict'])
    # xmax, xmin = load['xmax'], load['xmin']
    # ymax, ymin = load['ymax'], load['ymin']

    # # eval mode
    # model.eval()
