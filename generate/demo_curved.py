import gym
import particle_envs
import random
import numpy as np
import pickle as pkl
from pathlib import Path

fixel_goal = True
if fixel_goal:
    # fixed_goal
    num_goals = 1
    num_demos_per_goal = 1000
    save_name = f'curved_fixed_goal_{num_demos_per_goal}_demos'
else:
    # changing_goals
    num_goals = 1000
    num_demos_per_goal = 1
    save_name = f'curved_changing_goal_{num_goals}_demos'

# params
height = 200
width = 200
step_size = 10
seed = 101
save_image = False
save_dir = Path('./data/')

env = gym.make('particle-v0', height=height, width=width, step_size=step_size, 
               reward_type='dense', reward_scale=None, block=None)

# set seed
env.seed(seed)
np.random.seed(seed)

observations, states, actions, rewards, goals = [], [], [], [], []
incomplete = []

for i in range(num_goals):
    obs, st, act, rew, go = [], [], [], [], []
    for j in range(num_demos_per_goal):
        print("Episode: ", (i+1)*(j+1))
        # initialize lists
        o, s, a, r, g = [], [], [], [], []

        # Reset
        state = env.reset(reset_goal=(j==0))
        state = np.array(state).astype(np.int32)
        goal = env.goal
        g.append(np.array(goal).astype(np.float32))
        
        done = False
        step = 0
        
        # Generate random control points between start and end points
        start_point = np.array([state[0], state[1]])
        end_point = np.array([goal[0], goal[1]])

        direction = np.abs(end_point - start_point)
        if direction[0] > direction[1]:
            h1 = start_point[0]
            w1 = np.random.choice([1, -1]) * (end_point[1] - start_point[1]) + start_point[1]
            h2 = end_point[0]
            w2 = np.random.choice([1, -1]) * (end_point[1] - start_point[1]) + start_point[1]
        else:
            h1 = np.random.choice([1, -1]) * (end_point[0] - start_point[0]) + start_point[0]
            w1 = start_point[1]
            h2 = np.random.choice([1, -1]) * (end_point[0] - start_point[0]) + start_point[0]
            w2 = end_point[1]
        control_point1 = np.array([h1, w1])
        control_point2 = np.array([h2, w2])
        
        # Create parameter values
        num_steps = 31 if max(abs(goal[0]-state[0]), abs(goal[1]-state[1])) < 75 else 81
        t = np.linspace(0, 1, num=num_steps)

        # Compute the Bézier curve for x and y coordinates separately
        curve_x = (1 - t)**3 * start_point[0] + 3 * (1 - t)**2 * t * control_point1[0] + 3 * (1 - t) * t**2 * control_point2[0] + t**3 * end_point[0]
        curve_y = (1 - t)**3 * start_point[1] + 3 * (1 - t)**2 * t * control_point1[1] + 3 * (1 - t) * t**2 * control_point2[1] + t**3 * end_point[1]

        # Create the Bézier curve as a 2D array
        curve = np.column_stack((curve_x, curve_y))
        
        while not done:
            if np.linalg.norm(goal - state) < step_size:
                action = (goal - state) / step_size
            elif step >= (num_steps - 1):
                action = (goal - state) / np.linalg.norm(goal - state)
            else:
                action = (curve[step+1] - curve[step]) / step_size
            o.append(env.render(mode='rgb_array') if save_image else 0.)
            s.append(state.astype(np.float32))
            a.append(action.astype(np.float32))
            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state).astype(np.float32)
            r.append(float(reward))
            step += 1
            print("Step:", step, "State: ", state, "Action: ", action, "Next State: ", next_state, "Reward: ", reward, "Done: ", done, "Info: ", info)
            state = next_state
        o.append(env.render(mode='rgb_array') if save_image else 0.)
        s.append(state.astype(np.float32))
        a.append(np.zeros_like(action).astype(np.float32))
        r.append(float(reward))

        obs.append(np.array(o))
        st.append(np.array(s))
        act.append(np.array(a))
        rew.append(np.array(r))
        go.append(np.array(g))
        print("\n\n\n")

        # check if trajectory is incomplete
        if env.observation[int(state[0]), int(state[1])] < 2:
            incomplete.append((i+1)*(j+1))
    observations.extend(obs)
    states.extend(st)
    actions.extend(act)
    rewards.extend(rew)
    goals.extend(go)

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