import os
import urllib.request
from typing import cast

import expt
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import wandb.apis.reports as wb  # noqa
from expt import Hypothesis, Run
from expt.plot import GridPlot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from openrlbenchmark.atari_data import atari_human_normalized_scores


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


if not os.path.exists("seed_r2d2_atari_graphs.csv"):
    urllib.request.urlretrieve(
        "https://github.com/google-research/seed_rl/raw/66e8890261f09d0355e8bf5f1c5e41968ca9f02b/docs/seed_r2d2_atari_graphs.csv",
        "seed_r2d2_atari_graphs.csv",
    )

api = wandb.Api()

env_ids = atari_human_normalized_scores.keys()
y_names = ["Median human normalized score", "Mean human normalized score"]

NUM_SEEDS = 3
NUM_FRAME_STACK = 4
ours_median_runss = []
ours_mean_runss = []
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
            expt_run.df["_runtime"] = (
                expt_run.df["global_step"] * (0.75 / max(expt_run.df["global_step"])) * 60
            )  # normalize it 0.75 hours * 60 minutes per hour
        runss.extend(expt_runs)

    ex = expt.Experiment("Comparison of PPO")
    ex.add_runs("CleanRL's PPO", runss)
    figure = ex.plot(
        x="_runtime",
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
    ours_median_runss.extend(
        [
            Run(
                f"seed-{seed}",
                pd.DataFrame(
                    {
                        "_runtime": ax.lines[0].get_xdata(),
                        "median_human_normalized_score": ax.lines[0].get_ydata(),
                    },
                ),
            )
        ]
    )
    figure = ex.plot(
        x="_runtime",
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
    ours_mean_runss.extend(
        [
            Run(
                f"seed-{seed}",
                pd.DataFrame(
                    {
                        "_runtime": ax.lines[0].get_xdata(),
                        "mean_human_normalized_score": ax.lines[0].get_ydata(),
                    },
                ),
            )
        ]
    )


baselines_median_runss = []
baselines_mean_runss = []
for seed in range(1, NUM_SEEDS + 1):
    runss = []
    for env_id in env_ids:
        api = wandb.Api()
        wandb_runs = api.runs(
            path="openrlbenchmark/baselines",
            filters={
                "$and": [
                    {"config.env.value": env_id.replace("-v5", "NoFrameskip-v4")},
                    {"config.exp_name.value": "baselines-ppo2-cnn"},
                    {"config.seed.value": seed},
                ]
            },
        )
        expt_runs = create_expt_runs(wandb_runs)

        # normalize scores and adjust x-axis from steps to frames
        for expt_run in expt_runs:
            expt_run.df["charts/episodic_return"] = (
                expt_run.df["charts/episodic_return"] - atari_human_normalized_scores[env_id][0]
            ) / (atari_human_normalized_scores[env_id][1] - atari_human_normalized_scores[env_id][0])
            expt_run.df["global_step"] *= NUM_FRAME_STACK
            expt_run.df["_runtime"] = (
                expt_run.df["global_step"] * (0.75 / max(expt_run.df["global_step"])) * 140
            )  # normalize it 0.75 hours * 140 minutes per hour
        runss.extend(expt_runs)

    ex = expt.Experiment("Comparison of PPO")
    ex.add_runs("CleanRL's PPO", runss)
    figure = ex.plot(
        x="_runtime",
        y="charts/episodic_return",
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
    baselines_median_runss.extend(
        [
            Run(
                f"seed-{seed}",
                pd.DataFrame(
                    {
                        "_runtime": ax.lines[0].get_xdata(),
                        "median_human_normalized_score": ax.lines[0].get_ydata(),
                    },
                ),
            )
        ]
    )
    figure = ex.plot(
        x="_runtime",
        y="charts/episodic_return",
        rolling=5,
        n_samples=400,
        legend=False,
        err_fn=lambda h: h.grouped.sem(),
        err_style=None,
        title=y_names[1],
        suptitle="",
    )
    ax = figure.axes[0, 0]
    baselines_mean_runss.extend(
        [
            Run(
                f"seed-{seed}",
                pd.DataFrame(
                    {
                        "_runtime": ax.lines[0].get_xdata(),
                        "mean_human_normalized_score": ax.lines[0].get_ydata(),
                    },
                ),
            )
        ]
    )


g = GridPlot(y_names=y_names)
ex = expt.Experiment("Median human normalized scores")
ex.add_runs("CleanRL's PPO + JAX + EnvPool's XLA", ours_median_runss)
ex.add_runs("openai/baselines' PPO", baselines_median_runss)
ax = g[y_names[0]]
ex.plot(
    x="_runtime",
    y="median_human_normalized_score",
    rolling=5,
    n_samples=400,
    legend=False,
    # err_style="band",
    suptitle="",
    title=y_names[0],
    ax=ax,
)
ax.set_ylabel("")
# axins = inset_axes(ax, 1.8, 1, loc="lower right", bbox_to_anchor=(0.47, 0.25), bbox_transform=ax.figure.transFigure)
# ex.plot(
#     x="_runtime",
#     y="median_human_normalized_score",
#     rolling=5,
#     n_samples=400,
#     legend=False,
#     suptitle="",
#     # ax=axins,
# )
# axins.set_title("")
# axins.set_ylabel("")
# axins.set_xlabel("")
# axins.set_xlim(0, 60, auto=True)
# axins.set_ylim(0, 1.5, auto=True)
# axins.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


ex = expt.Experiment("mean human normalized scores")
ex.add_runs("CleanRL's PPO + JAX + EnvPool's XLA", ours_mean_runss)
ex.add_runs("openai/baselines' PPO", baselines_mean_runss)
ax = g[y_names[1]]
ex.plot(
    x="_runtime",
    y="mean_human_normalized_score",
    rolling=5,
    n_samples=400,
    legend=False,
    # err_style="band",
    suptitle="",
    title=y_names[1],
    ax=ax,
)
ax.set_ylabel("")
# axins = inset_axes(ax, 1.8, 1, loc="lower right", bbox_to_anchor=(0.95, 0.28), bbox_transform=ax.figure.transFigure)
# ex.plot(
#     x="_runtime",
#     y="mean_human_normalized_score",
#     rolling=5,
#     n_samples=400,
#     legend=False,
#     suptitle="",
#     # ax=axins,
# )
# axins.set_title("")
# axins.set_ylabel("")
# axins.set_xlabel("")
# axins.set_xlim(0, 60, auto=True)
# axins.set_ylim(0, 8, auto=True)
# axins.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


for ax in g:
    ax.yaxis.set_label_text("")
    ax.xaxis.set_label_text("Minutes")
handles, labels = ax.get_legend_handles_labels()
plt.figlegend(handles, labels, loc="lower center", bbox_to_anchor=[0.5, -0.15])
plt.savefig("static/hns_ppo_vs_baselines.png", bbox_inches="tight")
plt.savefig("static/hns_ppo_vs_baselines.svg", bbox_inches="tight")
