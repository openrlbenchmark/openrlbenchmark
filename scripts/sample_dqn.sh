#!/bin/bash
# 'sample_dqn?wandb_group=sample-dqn-cem-double-noise&cl=SampleDQN (double noise)' \

uv run python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=sbx&ceik=env&cen=algo&metric=eval/mean_reward' \
        'sample_dqn?wandb_group=sample-dqn-cem-triple-q&cl=SampleDQN (Triple Q)' \
        'sample_dqn?tag=sample-dqn-cem-single-q&cl=SampleDQN (Single Q)' \
        'ddpg?tag=ddpg-normalize&cl=DDPG' \
    --env-ids HalfCheetah-v4 Ant-v4 Hopper-v4 Walker2d-v4 Swimmer-v4 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 3 \
    --rliable \
    --rc.score_normalization_method maxmin \
    --rc.normalized_score_threshold 1.0 \
    --rc.sample_efficiency_plots \
    --rc.sample_efficiency_and_walltime_efficiency_method Median \
    --rc.performance_profile_plots  \
    --rc.aggregate_metrics_plots  \
    --rc.sample_efficiency_num_bootstrap_reps 1000 \
    --rc.performance_profile_num_bootstrap_reps 1000 \
    --rc.interval_estimates_num_bootstrap_reps 1000 \
    --rc.confidence_interval_size 0.95 \
    --output-filename static/sample_dqn/sample_dqn \
    --scan-history --offline
