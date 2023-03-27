import subprocess


def basic_test():
    subprocess.run(
        "python -m openrlbenchmark.rlops --filters '?we=openrlbenchmark&wpn=baselines&ceik=env&cen=exp_name&metric=charts/episodic_return' 'baselines-ppo2-cnn'  --filters '?we=openrlbenchmark&wpn=envpool-atari&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' 'ppo_atari_envpool_xla_jax_truncation'  --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' 'ppo_atari_envpool_xla_jax_scan?tag=pr-328'  --env-ids BeamRider-v5 Breakout-v5   --check-empty-runs True  --ncols 3  --ncols-legend 2  --output-filename compare  --scan-history",
        shell=True,
        check=True,
    )


