import subprocess


def test_plot_different_libraries():
    """
    test plotting against different libraries
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
    --pc.ncols 1 \
    --pc.ncols-legend 2 \
    --pc.xlabel 'Training Steps' \
    --pc.ylabel 'Episodic Return' \
    --output-filename static/0compare
""",
        shell=True,
        check=True,
    )
