import subprocess


def test_plot_different_libraries_scan_history():
    """
    same as above but with scan history, which caches runs
    """
    subprocess.run(
        """
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=sb3&ceik=env&cen=algo&metric=rollout/ep_rew_mean' \
        'a2c' \
        'ddpg' \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'sac_continuous_action?tag=rlops-pilot&cl=SAC' \
    --env-ids HalfCheetahBulletEnv-v0 \
    --no-check-empty-runs \
    --scan-history \
    --pc.ncols 1 \
    --pc.ncols-legend 2 \
    --pc.xlabel 'Training Steps' \
    --pc.ylabel 'Episodic Return' \
    --output-filename static/0compare
""",
        shell=True,
        check=True,
    )


def test_plot_different_libraries_and_env_ids():
    """
    each filter can have their own env ids
    """
    subprocess.run(
        """
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=envpool-atari&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' 'ppo_atari_envpool_xla_jax_truncation' \
    --filters '?we=openrlbenchmark&wpn=baselines&ceik=env&cen=exp_name&metric=charts/episodic_return' 'baselines-ppo2-cnn' \
    --env-ids Alien-v5 Amidar-v5 \
    --env-ids AlienNoFrameskip-v4 AmidarNoFrameskip-v4 \
    --no-check-empty-runs \
    --pc.ncols 5 \
    --pc.ncols-legend 2 \
    --output-filename static/0compare \
    --scan-history
""",
        shell=True,
        check=True,
    )


def test_plot_different_libraries_and_env_ids_offline():
    """
    testing offline db
    """
    subprocess.run(
        """
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=envpool-atari&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' 'ppo_atari_envpool_xla_jax_truncation' \
    --filters '?we=openrlbenchmark&wpn=baselines&ceik=env&cen=exp_name&metric=charts/episodic_return' 'baselines-ppo2-cnn' \
    --env-ids Alien-v5 Amidar-v5 \
    --env-ids AlienNoFrameskip-v4 AmidarNoFrameskip-v4 \
    --no-check-empty-runs \
    --pc.ncols 5 \
    --pc.ncols-legend 2 \
    --output-filename static/0compare \
    --scan-history --offline
""",
        shell=True,
        check=True,
    )


def test_rliable_hns():
    """
    test rliable hns integration
    """
    subprocess.run(
        """
python -m openrlbenchmark.rlops_hns \
    --filters '?we=openrlbenchmark&wpn=sb3&ceik=env&cen=algo&metric=eval/mean_reward' \
        'dqn?tag=v1.8.0a3&cl=DQN' \
        'a2c?tag=v1.8.0a3&cl=A2C' \
    --env-ids PongNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --no-check-empty-runs \
    --pc.ncols 1 \
    --pc.ncols-legend 2 \
    --output-filename static/0compare \
    --scan-history --rliable
""",
        shell=True,
        check=True,
    )


def test_params_filter():
    """
    test params filter
    """
    subprocess.run(
        """
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'ppo_continuous_action?tag=v1.0.0-27-gde3f410&seed=1&seed=2' \
    --env-ids HalfCheetah-v2 Walker2d-v2 Hopper-v2 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 3 \
    --output-filename static/0compare
""",
        shell=True,
        check=True,
    )
