from envs.stochastic_env import StochasticWorld
from exp_schedule import ExponentialSchedule
from dqn import train_dqn, plot, save
from expert_policy import ExpertPolicy
from tqdm import trange
import numpy as np

if __name__ == "__main__":
    env = StochasticWorld(500, 500, see_all=True)
    policy = ExpertPolicy(env)
    gamma = 0.99

    steps = 5_000
    env.reset(render=True)
    G = 0
    done = False
    timestep = 0
    returns = []
    lengths = []
    losses = []
    for step in trange(steps, desc="steps"):
        # take a step
        action = policy()
        next_state, reward, done = env.step(action, render=True)
        timestep += 1
        G = reward + gamma * G
        if done:
            print("discounted return", G, "timestep", timestep)
            env.reset(render=True)
            returns.append(G)
            lengths.append(timestep)
            done = False
            G = 0
            timestep = 0

    save("results/point_and_shoot.txt", returns, lengths, losses)
    plot(np.array(returns), np.array(lengths), np.array(losses))