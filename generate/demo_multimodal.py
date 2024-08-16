import gym
import particle_envs
import numpy as np
import pickle as pkl
from pathlib import Path

num_demos = 1000
height = 200
width = 200 
step_size = 5
seed = 101
save_image = False
save_dir = Path('./data/')
save_name = f'multimodal_{num_demos}_demos'

env = gym.make('particle-v0', height=height, width=width, step_size=step_size, 
               reward_type='dense', reward_scale=None, block=None)

start_state = [5,5]
goal_state = [height-1, width-1]
goal_vary_range = 5
# Paths
paths = [
    [[height//2, 5],
    [height//2, width//2],
    [height//2, width-1],
    goal_state],
    
    [[5, width//2],
    [height//2, width//2],
    [height-1, width//2],
    goal_state],

    [[height//2, 5],
    [height//2, width//2],
    [height-1, width//2],
    goal_state],

    [[5, width//2],
    [height//2, width//2],
    [height//2, width-1],
    goal_state]
]
start_state = np.array(start_state).astype(np.float32)
goal_state = np.array(goal_state).astype(np.float32)
paths = np.array(paths).astype(np.float32)

# set seed
env.seed(seed)
np.random.seed(seed)

observations, states, actions, rewards, goals = [], [], [], [], []
incomplete = []


for i in range(num_demos//len(paths)):
    obs, st, act, rew, gs = [], [], [], [], []
    for j in range(len(paths)):
        print("Episode: ", (i+1)*(j+1))
        # initialize lists
        o, s, a, r, g = [], [], [], [], []

        # Reset
        state = env.reset(start_state=start_state, reset_goal=True, goal_state=goal_state)
        state = np.array([state[0] * height, state[1] * width]).astype(np.int32)
        
        #ntermediate goals
        goal_idx = 0
        goal = paths[j][goal_idx]
        
        done = False
        step = 0
        while not done:
            g.append(np.array(goal).astype(np.float32))
            if np.linalg.norm(goal - state) < step_size:
                action = (goal - state) / step_size
                # update to next goal
                goal_idx += 1
                if goal_idx < len(paths[j])-1:
                    goal = paths[j][goal_idx] + [np.random.randint(-goal_vary_range, goal_vary_range), np.random.randint(-goal_vary_range, goal_vary_range)]
                    goal = np.clip(goal, 0, [height-1, width-1])
                elif goal_idx == len(paths[j])-1:
                    goal = paths[j][goal_idx]
            else:
                action = (goal - state) / np.linalg.norm(goal - state)
                # action = unit_vector * step_size
            o.append(env.render(mode='rgb_array') if save_image else 0.)
            s.append(state.astype(np.float32))
            a.append(action.astype(np.float32))
            next_state, reward, done, info = env.step(action)
            next_state = np.array([next_state[0], next_state[1]]).astype(np.float32)
            r.append(float(reward))
            step += 1
            print("Step:", step, "State: ", state, "Action: ", action, "Next State: ", next_state, "Reward: ", reward, "Done: ", done, "Info: ", info)
            state = next_state
        o.append(env.render(mode='rgb_array') if save_image else 0.)
        s.append(state.astype(np.float32))
        a.append(np.zeros_like(action).astype(np.float32))
        r.append(float(reward))
        g.append(np.array(goal).astype(np.float32))

        obs.append(o)
        st.append(s)
        act.append(a)
        rew.append(r)
        gs.append(g)
        print("\n\n\n")

        # check if trajectory is incomplete
        if env.observation[int(state[0]), int(state[1])] < 2:
            incomplete.append((i+1)*(j+1))
    
    observations.extend(obs)
    states.extend(st)
    actions.extend(act)
    rewards.extend(rew)
    goals.extend(gs)

# Save data
save_dir.mkdir(parents=True, exist_ok=True)
with open(save_dir / (save_name + '.pkl'), 'wb') as f:
    pkl.dump({
        'observations': observations,
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'goals': goals
    }, f)

# Print incomplete
print("Incomplete: ", len(incomplete))
for el in incomplete:
    print(el)