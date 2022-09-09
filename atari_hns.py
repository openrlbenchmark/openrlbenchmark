import expt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import wandb.apis.reports as wb  # noqa
from expt import Run
from expt.plot import GridPlot

from atari_data import atari_human_normalized_scores


class Runset:
    def __init__(self, name: str, filters: dict, entity: str, project: str, groupby: str = ""):
        self.name = name
        self.filters = filters
        self.entity = entity
        self.project = project
        self.groupby = groupby

    @property
    def runs(self):
        return wandb.Api().runs(path=f"{self.entity}/{self.project}", filters=self.filters)

    @property
    def report_runset(self):
        return wb.RunSet(
            name=self.name,
            entity=self.entity,
            project=self.project,
            filters={"$or": [self.filters]},
            groupby=[self.groupby] if len(self.groupby) > 0 else None,
        )


def create_expt_runs(wandb_runs):
    runs = []
    for idx, run in enumerate(wandb_runs):
        wandb_run = run.history()
        if "videos" in wandb_run:
            wandb_run = wandb_run.drop(columns=["videos"], axis=1)
        runs += [Run(f"seed{idx}", wandb_run)]
    return runs


api = wandb.Api()
wandb.require("report-editing")

env_ids = atari_human_normalized_scores.keys()
hms = []
raw_scores = []
NUM_FRAME_STACK = 4
NUM_COLS = 5
g = GridPlot(y_names=env_ids, layout=(int(np.ceil(len(env_ids) / NUM_COLS)), NUM_COLS))
blocks = []

for env_id in env_ids:

    runset1 = Runset(
        name="CleanRL ppo_atari_envpool_xla_jax.py",
        filters={"$and": [{"config.env_id.value": env_id}, {"config.exp_name.value": "ppo_atari_envpool_xla_jax"}]},
        entity="openrlbenchmark",
        project="envpool-atari",
        groupby="exp_name",
    )
    blocks += [
        wb.PanelGrid(
            runsets=[
                runset1.report_runset,
            ],
            panels=[
                wb.LinePlot(
                    x="global_step",
                    y=["charts/avg_episodic_return"],
                    title=env_id,
                    title_x="Steps",
                    title_y="Episodic Return",
                    max_runs_to_show=100,
                    smoothing_factor=0.8,
                    groupby_rangefunc="stderr",
                    legend_template="${runsetName}",
                ),
                wb.LinePlot(
                    x="_runtime",
                    y=["charts/avg_episodic_return"],
                    title=env_id,
                    title_y="Episodic Return",
                    max_runs_to_show=100,
                    smoothing_factor=0.8,
                    groupby_rangefunc="stderr",
                    legend_template="${runsetName}",
                ),
            ],
        ),
    ]

    wandb_runs = api.runs(
        path="openrlbenchmark/envpool-atari",
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

    ex.add_runs("CleanRL's PPO + JAX + EnvPool's XLA", expt_runs)
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
    raw_scores += [ex.summary()["charts/avg_episodic_return"][0]]
    hms += [ex.summary()["human normalized score"][0]]

g.add_legend(ax=g.axes[-1, -1], loc="upper left", bbox_to_anchor=(0, 1))
for ax in g.axes_active:
    ax.xaxis.set_label_text("")
    ax.yaxis.set_label_text("")

plt.tight_layout(w_pad=1)
plt.savefig("static/hms_each_game.png")
plt.savefig("static/hms_each_game.svg")

pd.DataFrame(sorted(zip(env_ids, raw_scores, hms)), columns=["Environment", "Return", "Human Normalized Score"]).to_markdown(
    "static/atari_hns.md"
)
plt.clf()
plt.rcdefaults()
sorted_tuple = sorted(zip(hms, env_ids))
sorted_hms = [x for x, _ in sorted_tuple]
sorted_env_ids = [x for _, x in sorted_tuple]

fig, ax = plt.subplots(figsize=(7, 10))
y_pos = np.arange(len(sorted_env_ids))
bars = ax.barh(y_pos, sorted_hms)
ax.bar_label(bars, fmt="%.2f")
ax.set_yticks(y_pos, labels=sorted_env_ids)
plt.tight_layout()
plt.savefig("static/hms_bar.png")
plt.savefig("static/hms_bar.svg")

report = wb.Report(
    project="openrlbenchmark",
    entity="openrlbenchmark",
    title="Atari: CleanRL PPO + JAX + EnvPool's XLA (part 1)",
    blocks=blocks[:29],
)
report.save()
print(f"view the generated report at {report.url}")
report = wb.Report(
    project="openrlbenchmark",
    entity="openrlbenchmark",
    title="Atari: CleanRL PPO + JAX + EnvPool's XLA (part 2)",
    blocks=blocks[29:],
)
report.save()
print(f"view the generated report at {report.url}")
