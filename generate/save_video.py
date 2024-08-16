import sys
sys.path.append('../')

import cv2
import imageio
import numpy as np
from pathlib import Path
import pickle as pkl

class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(obs,
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)


names = [
    'straight_fixed_goal_1000_demos'
]
'''    'straight_changing_goal_10_demos',
    'multimodal_10_demos','''

for name in names:
    DEMO_PATH = Path(f"./data/{name}.pkl")
    SAVE_DIR = Path("./data/")

    with open(DEMO_PATH, "rb") as f:
        data = pkl.load(f)
        demos = data["observations"]
        goals = data["goals"]

    recorder = TrainVideoRecorder(SAVE_DIR)

    num_demos = len(demos)

    recorder.init(demos[0][0])
    for i in range(num_demos):
        for frame in demos[i]:
            recorder.record(frame)
    recorder.save(f'demo_{name}.mp4')
