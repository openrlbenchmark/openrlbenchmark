import subprocess


def test_plot_different_libraries():
    """
    test plotting against different libraries
    """
    subprocess.run("""
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=sb3&ceik=env&cen=algo&metric=rollout/ep_rew_mean' \
        'a2c' \
        'ddpg' \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'sac_continuous_action?tag=rlops-pilot&cl=SAC' \
    --env-ids HalfCheetahBulletEnv-v0 \
    --ncols 1 \
    --ncols-legend 2 \
    --xlabel 'Training Steps' \
    --ylabel 'Episodic Return' \
    --output-filename static/0compare
""",
        shell=True,
        check=True,
    )


def test_plot_different_libraries_scan_hisotry():
    """
    same as above but with scan history, which caches runs
    """
    subprocess.run("""
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=sb3&ceik=env&cen=algo&metric=rollout/ep_rew_mean' \
        'a2c' \
        'ddpg' \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'sac_continuous_action?tag=rlops-pilot&cl=SAC' \
    --env-ids HalfCheetahBulletEnv-v0 \
    --no-check-empty-runs \
    --scan-history \
    --ncols 1 \
    --ncols-legend 2 \
    --xlabel 'Training Steps' \
    --ylabel 'Episodic Return' \
    --output-filename static/0compare
""",
        shell=True,
        check=True,
    )


def test_plot_different_libraries_and_env_ids():
    """
    each filter can have their own env ids
    """
    subprocess.run("""
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=envpool-atari&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' 'ppo_atari_envpool_xla_jax_truncation' \
    --filters '?we=openrlbenchmark&wpn=baselines&ceik=env&cen=exp_name&metric=charts/episodic_return' 'baselines-ppo2-cnn' \
    --env-ids Alien-v5 Amidar-v5 \
    --env-ids AlienNoFrameskip-v4 AmidarNoFrameskip-v4 \
    --no-check-empty-runs \
    --ncols 5 \
    --ncols-legend 2 \
    --output-filename static/0compare \
    --scan-history
""",
        shell=True,
        check=True,
    )


def test_plot_different_libraries_and_env_ids_offline():
    """
    each filter can have their own env ids
    """
    subprocess.run("""
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=envpool-atari&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' 'ppo_atari_envpool_xla_jax_truncation' \
    --filters '?we=openrlbenchmark&wpn=baselines&ceik=env&cen=exp_name&metric=charts/episodic_return' 'baselines-ppo2-cnn' \
    --env-ids Alien-v5 Amidar-v5 \
    --env-ids AlienNoFrameskip-v4 AmidarNoFrameskip-v4 \
    --no-check-empty-runs \
    --ncols 5 \
    --ncols-legend 2 \
    --output-filename static/0compare \
    --scan-history --offline
""",
        shell=True,
        check=True,
    )


