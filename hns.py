from typing import cast

import expt
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import wandb.apis.reports as wb  # noqa
from expt import Hypothesis, Run
from expt.plot import GridPlot

from atari_data import atari_human_normalized_scores


def repr_fn(h: Hypothesis) -> pd.DataFrame:
    # A dummy function that manipulates the representative value ('median')
    df = cast(pd.DataFrame, h.grouped.median())
    return df


def create_expt_runs(wandb_runs):
    runs = []
    for idx, run in enumerate(wandb_runs):
        wandb_run = run.history()
        if "videos" in wandb_run:
            wandb_run = wandb_run.drop(columns=["videos"], axis=1)
        runs += [Run(f"seed{idx}", wandb_run)]
    return runs


api = wandb.Api()

env_ids = atari_human_normalized_scores.keys()
y_names = ["Median human normalized score", "Mean human normalized score"]

NUM_SEEDS = 3
NUM_FRAME_STACK = 4
median_runss = []
mean_runss = []
for seed in range(1, NUM_SEEDS + 1):
    runss = []
    for env_id in env_ids:
        api = wandb.Api()
        wandb_runs = api.runs(
            path="openrlbenchmark/envpool-atari",
            filters={
                "$and": [
                    {"config.env_id.value": env_id},
                    {"config.exp_name.value": "ppo_atari_envpool_xla_jax"},
                    {"config.seed.value": seed},
                ]
            },
        )
        expt_runs = create_expt_runs(wandb_runs)

        # normalize scores and adjust x-axis from steps to frames
        for expt_run in expt_runs:
            expt_run.df["charts/avg_episodic_return"] = (
                expt_run.df["charts/avg_episodic_return"] - atari_human_normalized_scores[env_id][0]
            ) / (atari_human_normalized_scores[env_id][1] - atari_human_normalized_scores[env_id][0])
            expt_run.df["global_step"] *= NUM_FRAME_STACK
        runss.extend(expt_runs)

    ex = expt.Experiment("Comparison of PPO")
    ex.add_runs("CleanRL's PPO", runss)
    figure = ex.plot(
        x="global_step",
        y="charts/avg_episodic_return",
        rolling=5,
        n_samples=400,
        legend=False,
        err_fn=None,
        err_style=None,
        suptitle="",
        title=y_names[0],
        representative_fn=repr_fn,
    )
    ax = figure.axes[0, 0]
    median_runss.extend(
        [
            Run(
                f"seed-{seed}",
                pd.DataFrame(
                    {
                        "global_step": ax.lines[0].get_xdata(),
                        "median_human_normalized_score": ax.lines[0].get_ydata(),
                    },
                ),
            )
        ]
    )
    figure = ex.plot(
        x="global_step",
        y="charts/avg_episodic_return",
        rolling=5,
        n_samples=400,
        legend=False,
        err_fn=lambda h: h.grouped.sem(),
        err_style=None,
        title=y_names[1],
        suptitle="",
    )
    ax = figure.axes[0, 0]
    mean_runss.extend(
        [
            Run(
                f"seed-{seed}",
                pd.DataFrame(
                    {
                        "global_step": ax.lines[0].get_xdata(),
                        "mean_human_normalized_score": ax.lines[0].get_ydata(),
                    },
                ),
            )
        ]
    )

g = GridPlot(y_names=y_names)
ex = expt.Experiment("Median human normalized scores")
ex.add_runs("CleanRL's PPO", median_runss)
ex.plot(
    x="global_step",
    y="median_human_normalized_score",
    rolling=5,
    n_samples=400,
    legend=False,
    # err_style="band",
    suptitle="",
    title=y_names[0],
    ax=g[y_names[0]],
)

ex = expt.Experiment("mean human normalized scores")
ex.add_runs("CleanRL's PPO", mean_runss)
ex.plot(
    x="global_step",
    y="mean_human_normalized_score",
    rolling=5,
    n_samples=400,
    legend=False,
    # err_style="band",
    suptitle="",
    title=y_names[1],
    ax=g[y_names[1]],
)
for ax in g:
    ax.yaxis.set_label_text("")
    ax.xaxis.set_label_text("Frames")
plt.savefig("hns.png", bbox_inches="tight")
