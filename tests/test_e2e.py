import subprocess


def test_plot_different_libraries_scan_hisotry():
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
python -i -m openrlbenchmark.rlops_hns \
    --filters '?we=costa-huang&wpn=moolib-atari-2&ceik=env_id&cen=exp_name&metric=global/mean_episode_return' \
        'moolib_impala_envpool_machado?cl=Moolib (Resnet CNN) 1 A100, 10 CPU' \
    --filters '?we=openrlbenchmark&wpn=moolib-atari&ceik=env_id&cen=exp_name&metric=global/mean_episode_return' \
        'moolib_impala_envpool_machado_8gpu_actor_batch_size16?cl=Moolib (Resnet CNN) 8 A100, 80 CPU'  \
    --filters '?we=openrlbenchmark&wpn=cleanba&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
        'cleanba_impala_envpool_machado_atari_wrapper_a0_l0_d1_nmb4?tag=v0.0.1-28-gdc44d45&cl=Cleanba IMPALA (Resnet CNN), 1 A100, 10 CPU' \
        'cleanba_impala_envpool_machado_atari_wrapper_a0_l1_d4?tag=v0.0.1-31-gb5e05f8&cl=Cleanba IMPALA (Resnet CNN) 8 A100, 50 CPU' \
        'cleanba_ppo_envpool_machado_atari_wrapper_a0_l0_d1_cpu10?tag=v0.0.1-28-gdc44d45&cl=Cleanba PPO (Resnet CNN), 1 A100, 10 CPU' \
        'cleanba_ppo_envpool_machado_atari_wrapper?tag=v0.0.1-16-g32dbf31&cl=Cleanba PPO (Resnet CNN) 8 A100, 50 CPU' \
    --env-ids Alien-v5 Amidar-v5 \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 2 \
    --output-filename static/0compare \
    --scan-history --offline --rliable
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
