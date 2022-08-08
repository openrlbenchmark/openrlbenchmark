import expt
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import wandb.apis.reports as wb  # noqa
from expt import Hypothesis, Run
from expt.plot import GridPlot
import pandas as pd
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
y_names = ["Median human normalized score", "Median human normalized score 2", "Mean human normalized score"]

df = pd.read_csv("seed_r2d2_atari_graphs.csv")
df = df.set_index(['game', 'seed'])
NUM_FRAME_STACK = 4
runss = []
for env_id in env_ids:
    if env_id in ["Surround-v5", "Defender-v5"]:
        continue
    expt_runs = [Run(f"{env_id}_seed0", df.loc[env_id.replace("-v5", ""), 0].reset_index())]

    # normalize scores and adjust x-axis from steps to frames
    for expt_run in expt_runs:
        expt_run.df["charts/avg_episodic_return"] = (
            expt_run.df["return"] - atari_human_normalized_scores[env_id][0]
        ) / (atari_human_normalized_scores[env_id][1] - atari_human_normalized_scores[env_id][0])
        expt_run.df["global_step"] = expt_run.df["training_step"]
    runss.extend(expt_runs)


from typing import cast


def repr_fn(h: Hypothesis) -> pd.DataFrame:
    # A dummy function that manipulates the representative value ('median')
    df = cast(pd.DataFrame, h.grouped.median())
    # df['loss'] = np.asarray(df.reset_index()['step']) * -1.0
    return df


g = GridPlot(y_names=y_names)
ex = expt.Experiment("Comparison of PPO")
ex.add_runs("SeedRL's R2D2", runss)

runss = []
for env_id in env_ids:
    api = wandb.Api()
    wandb_runs = api.runs(
        path="costa-huang/envpool-atari",
        filters={"$and": [{"config.env_id.value": env_id}, {"config.exp_name.value": "ppo_atari_envpool_xla_jax"}]},
    )
    expt_runs = create_expt_runs(wandb_runs)

    # normalize scores and adjust x-axis from steps to frames
    for expt_run in expt_runs:
        expt_run.df["charts/avg_episodic_return"] = (
            expt_run.df["charts/avg_episodic_return"] - atari_human_normalized_scores[env_id][0]
        ) / (atari_human_normalized_scores[env_id][1] - atari_human_normalized_scores[env_id][0])
        expt_run.df["global_step"] *= NUM_FRAME_STACK
    runss.extend(expt_runs)

ex.add_runs("CleanRL's PPO", runss)
ex.plot(
    x="global_step",
    y="charts/avg_episodic_return",
    rolling=50,
    n_samples=400,
    legend=False,
    err_fn=None,
    err_style=None,
    suptitle="",
    title=y_names[0],
    representative_fn=repr_fn,
    ax=g[y_names[0]],
)
ex.plot(
    x="global_step",
    y="charts/avg_episodic_return",
    rolling=50,
    n_samples=400,
    legend=False,
    err_fn=None,
    # err_style="fill",
    suptitle="",
    title=y_names[1],
    representative_fn=repr_fn,
    ax=g[y_names[1]],
)
ex.plot(
    x="global_step",
    y="charts/avg_episodic_return",
    rolling=50,
    n_samples=400,
    legend=False,
    err_fn=lambda h: h.grouped.sem(),
    err_style="fill",
    title=y_names[2],
    suptitle="",
    ax=g[y_names[2]],
)
for ax in g:
    ax.yaxis.set_label_text("")
    ax.xaxis.set_label_text("Frames")
plt.savefig("hns_ppo_vs_r2d2.png", bbox_inches="tight")
