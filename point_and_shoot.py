from envs.obstacles_env import ObstaclesWorld
from exp_schedule import ExponentialSchedule
from dqn import train_dqn, plot, save
from expert_policy import ExpertPolicy
from tqdm import trange

if __name__ == "__main__":
    env = ObstaclesWorld(500, 500, see_all=True)
    policy = ExpertPolicy(env)
    gamma = 0.99
    

    
    steps = 50_000
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
        if (done): 
            print('discounted return', G, 'timestep', timestep)
            env.reset(render=True)
            returns.append(G)
            lengths.append(timestep)
            done = False
            G = 0
            timestep = 0
            

    plot(returns, lengths, losses)
    save("dqn_completed.txt", returns, lengths, losses)
    # assert len(dqn_models) == num_saves
    # assert all(isinstance(value, DQN) for value in dqn_models.values())

    # saving computed models to disk, so that we can load and visualize them later.
    # checkpoint = {key: dqn.custom_dump() for key, dqn in dqn_models.items()}
    # torch.save(checkpoint, f'checkpoint_{env.spec.id}.pt')