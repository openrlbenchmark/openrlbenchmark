from openrlbenchmark import plot_atari, Runset
# from openrlbenchmark.atari_data import atari_human_normalized_scores

plot_atari.plot_atari([
    Runset(
        name="CleanRL ppo_atari_envpool_xla_jax.py",
        filters=[{"config.exp_name.value": "ppo_atari_envpool_xla_jax"}],
        entity="openrlbenchmark",
        project="envpool-atari",
        groupby="exp_name",
        key_for_env_id="config.env_id.value",
        x_axis="global_step",
        y_axis="charts/avg_episodic_return",
    ),
    Runset(
        name="openai/baselines' PPO",
        filters=[{"config.exp_name.value": "baselines-ppo2-cnn"}],
        entity="openrlbenchmark",
        project="baselines",
        groupby="exp_name",
        key_for_env_id="config.env.value",
        x_axis="global_step",
        y_axis="charts/episodic_return",
        env_id_fn=lambda env_id: env_id.replace("-v5", "NoFrameskip-v4"),
    ),
])