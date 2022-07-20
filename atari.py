import expt
import matplotlib.pyplot as plt
import wandb
import wandb.apis.reports as wb  # noqa
from expt import Hypothesis, Run
from expt.plot import GridPlot

from atari_data import atari_human_normalized_scores


def create_hypothesis(name, wandb_runs):
    runs = []
    for idx, run in enumerate(wandb_runs):
        wandb_run = run.history()
        if "videos" in wandb_run:
            wandb_run = wandb_run.drop(columns=["videos"], axis=1)
        runs += [Run(f"seed{idx}", wandb_run)]
    return Hypothesis(name, runs)


env_ids = atari_human_normalized_scores.keys()
# env_ids = ["Ant-v4", "HalfCheetah-v4"]

g = GridPlot(y_names=env_ids)

for env_id in env_ids:
    api = wandb.Api()
    wandb_runs = api.runs(
        path="costa-huang/envpool-atari",
        filters={"$and": [{"config.env_id.value": env_id}, {"config.exp_name.value": "ppo_atari_envpool_xla_jax"}]},
    )
    ex = expt.Experiment("Comparison of PPO")
    h = create_hypothesis("CleanRL's PPO + Envpool", wandb_runs)
    ex.add_hypothesis(h)

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

g.add_legend(ax=g.axes[-1, -1], loc="upper left", bbox_to_anchor=(0, 1))

# Some post-processing with matplotlib API so that the plot looks nicer
for ax in g.axes_active:
    ax.xaxis.set_label_text("")
    ax.yaxis.set_label_text("")
    # ax.set_xlim(0, 200e6)

# g.fig.tight_layout(h_pad=1.3, w_pad=1.0)

plt.savefig("test.png")

# r = wandb_runs[0].history()
