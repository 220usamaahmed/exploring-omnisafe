import omnisafe


env_id = "SafetyPointGoal1-v0"
custom_cfgs = {
    "train_cfgs": {
        "total_steps": 2048,
        "vector_env_nums": 1,
        "parallel": 1,
    },
    "algo_cfgs": {
        "steps_per_epoch": 1024,
        "update_iters": 1,
    },
    "logger_cfgs": {
        "use_wandb": False,
    },
}

agent = omnisafe.Agent("DDPG", env_id, custom_cfgs=custom_cfgs)
agent.learn()

agent.render(num_episodes=1, render_mode="rgb_array", width=256, height=256)
