import expt
import matplotlib.pyplot as plt
import numpy as np
import wandb
import wandb.apis.reports as wb  # noqa
from expt import Run
from expt.plot import GridPlot

from atari_data import atari_human_normalized_scores


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
hms = []
NUM_FRAME_STACK = 4
g = GridPlot(y_names=env_ids)


for env_id in env_ids:
    api = wandb.Api()
    wandb_runs = api.runs(
        path="costa-huang/envpool-atari",
        filters={"$and": [{"config.env_id.value": env_id}, {"config.exp_name.value": "ppo_atari_envpool_xla_jax"}]},
    )
    ex = expt.Experiment("Comparison of PPO")
    expt_runs = create_expt_runs(wandb_runs)

    # normalize scores and adjust x-axis from steps to frames
    for expt_run in expt_runs:
        expt_run.df["human normalized score"] = (
            expt_run.df["charts/avg_episodic_return"] - atari_human_normalized_scores[env_id][0]
        ) / (atari_human_normalized_scores[env_id][1] - atari_human_normalized_scores[env_id][0])
        expt_run.df["global_step"] *= NUM_FRAME_STACK

    ex.add_runs("CleanRL's PPO", expt_runs)
    ex.plot(
        ax=g[env_id],
        title=env_id,
        x="_runtime",
        y="charts/avg_episodic_return",
        err_style="band",
        std_alpha=0.1,
        rolling=50,
        n_samples=400,
        legend=False,
    )
    ax = g[env_id]
    ax2 = ax.twinx()
    ax2.set_ylim([0, ex.summary()["human normalized score"][0]])
    hms += [ex.summary()["human normalized score"][0]]

g.add_legend(ax=g.axes[-1, -1], loc="upper left", bbox_to_anchor=(0, 1))
for ax in g.axes_active:
    ax.xaxis.set_label_text("")
    ax.yaxis.set_label_text("")

plt.tight_layout(w_pad=1)
plt.savefig("test.png")


plt.clf()
plt.rcdefaults()
sorted_tuple = sorted(zip(hms, env_ids))
hms = [x for x, _ in sorted_tuple]
env_ids = [x for _, x in sorted_tuple]

fig, ax = plt.subplots(figsize=(7, 10))
y_pos = np.arange(len(env_ids))
bars = ax.barh(y_pos, hms)
ax.bar_label(bars, fmt="%.2f")
ax.set_yticks(y_pos, labels=env_ids)
plt.tight_layout()
plt.savefig("hms_bar.png")
