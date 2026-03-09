#!/bin/bash
# 'sample_dqn?wandb_group=sample-dqn-cem-double-noise&cl=SampleDQN (double noise)' \

uv run python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=sbx&ceik=env&cen=algo&metric=eval/mean_reward' \
        'sample_dqn?wandb_group=sample-dqn-cem-triple-q&cl=SampleDQN (Triple Q)' \
        'sample_dqn?tag=sample-dqn-cem-single-q&cl=SampleDQN (Single Q)' \
        'ddpg?tag=ddpg-normalize&cl=DDPG' \
    --filters '?we=openrlbenchmark&wpn=sb3&ceik=env&cen=algo&metric=eval/mean_reward' \
        'sac?cl=SAC (SB3)' \
        'td3?cl=TD3 (SB3)' \
    --env-ids HalfCheetah-v4 Ant-v4 Hopper-v4 Walker2d-v4 Swimmer-v4 Humanoid-v4 \
    --env-ids HalfCheetah-v3 Ant-v3 Hopper-v3 Walker2d-v3 Swimmer-v3 Humanoid-v3 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 3 \
    --rliable \
    --rc.score_normalization_method mujoco \
    --rc.normalized_score_threshold 1.0 \
    --rc.sample_efficiency_plots \
    --rc.sample_efficiency_and_walltime_efficiency_method IQM \
    --rc.performance_profile_plots  \
    --rc.aggregate_metrics_plots  \
    --rc.sample_efficiency_num_bootstrap_reps 1000 \
    --rc.performance_profile_num_bootstrap_reps 1000 \
    --rc.interval_estimates_num_bootstrap_reps 1000 \
    --rc.confidence_interval_size 0.95 \
    --output-filename static/sample_dqn/sample_dqn \
    --scan-history
