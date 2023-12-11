from envs.obstacles_env import ObstaclesWorld
from exp_schedule import ExponentialSchedule
from dqn import train_dqn, plot, save
from expert_policy import ExpertPolicy

if __name__ == "__main__":
    env = ObstaclesWorld(500, 500, see_all=True)
    gamma = 0.99

    # we train for many time-steps;  as usual, you can decrease this during development / debugging.
    # but make sure to restore it to 1_500_000 before submitting.
    num_steps = 50_000
    num_saves = 5  # save models at 0%, 25%, 50%, 75% and 100% of training

    replay_size = 70_000
    replay_prepopulate_steps = 50_000 #10 #50_000

    batch_size = 64
    exploration = ExponentialSchedule(0.3, 0.01, 30_000)
    

    # this should take about 90-120 minutes on a generic 4-core laptop
    dqn_models, returns, lengths, losses = train_dqn(
        env,
        observation_space=63,
        action_space=3,
        num_steps=num_steps,
        num_saves=num_saves,
        replay_size=replay_size,
        replay_prepopulate_steps=replay_prepopulate_steps,
        replay_policy=ExpertPolicy(env),
        batch_size=batch_size,
        exploration=exploration,
        gamma=gamma,
        render=True
    )

    plot(returns, lengths, losses)
    save("dqn_completed.txt", returns, lengths, losses)
    # assert len(dqn_models) == num_saves
    # assert all(isinstance(value, DQN) for value in dqn_models.values())

    # saving computed models to disk, so that we can load and visualize them later.
    # checkpoint = {key: dqn.custom_dump() for key, dqn in dqn_models.items()}
    # torch.save(checkpoint, f'checkpoint_{env.spec.id}.pt')