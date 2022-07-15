import enum
from matplotlib import axis
import numpy as np
import expt
from expt import Run, Hypothesis, Experiment
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import wandb.apis.reports as wb  # noqa
from expt.plot import GridPlot

def create_hypothesis(name, wandb_runs):
    runs = []
    for idx, run in enumerate(wandb_runs):
        wandb_run = run.history()
        if 'videos' in wandb_run:
            wandb_run = wandb_run.drop(columns=['videos'], axis=1)
        runs += [Run(f"seed{idx}", wandb_run)]
    return Hypothesis(name, runs)

env_ids = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "InvertedDoublePendulum-v4", "InvertedPendulum-v4", "Pusher-v4", "Reacher-v4", "Swimmer-v4", "Walker2d-v4",]
# env_ids = ["Ant-v4", "HalfCheetah-v4"]

g = GridPlot(y_names=env_ids)

for env_id in env_ids:
    api = wandb.Api()
    wandb_runs = api.runs(
        path="openrlbenchmark/envpool-cleanrl",
        filters={'$and': [{'config.env_id.value': env_id}, {'config.exp_name.value': 'ppo_continuous_action_envpool'}]}
    )
    wandb_runs2 = api.runs(
        path="openrlbenchmark/baselines",
        filters={'$and': [{'config.env.value': env_id.replace("v4", "v2")}, {'config.exp_name.value': 'baselines-ppo2-mlp'}]}
    )
    ex = expt.Experiment("Comparison of PPO")
    h = create_hypothesis("CleanRL's PPO + Envpool", wandb_runs)
    ex.add_hypothesis(h)
    h2 = create_hypothesis("openai/baselines' PPO", wandb_runs2)
    ex.add_hypothesis(h2)
    # ex.plot(ax=g[env_id], title=env_id, y=, x="_runtime", n_samples=300, rolling=50, err_style="fill", tight_layout=False, legend=False)

    ex.plot(ax=g[env_id], title=env_id,
            x='_runtime', y="charts/episodic_return",
            err_style='band', std_alpha=0.1,
            rolling=50, n_samples=400, legend=False,
            # tight_layout=False,
           )

g.add_legend(ax=g.axes[-1, -1], loc="upper left", bbox_to_anchor=(0, 1))

# Some post-processing with matplotlib API so that the plot looks nicer
for ax in g.axes_active:
    ax.xaxis.set_label_text("")
    ax.yaxis.set_label_text("")
    # ax.set_xlim(0, 200e6)

# g.fig.tight_layout(h_pad=1.3, w_pad=1.0)

plt.savefig("test.png")

# r = wandb_runs[0].history()