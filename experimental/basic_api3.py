from typing import List

import expt
import matplotlib.pyplot as plt
import numpy as np
import wandb
import wandb.apis.reports as wb  # noqa
from expt import Hypothesis, Run

wandb.require("report-editing")
api = wandb.Api()


def create_hypothesis(name: str, wandb_runs: List[wandb.apis.public.Run]) -> Hypothesis:
    runs = []
    for idx, run in enumerate(wandb_runs):
        wandb_run = run.history()
        if "videos" in wandb_run:
            wandb_run = wandb_run.drop(columns=["videos"], axis=1)
        runs += [Run(f"seed{idx}", wandb_run)]
    return Hypothesis(name, runs)


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


def compare(
    runsetss: List[List[Runset]],
    env_ids: List[str],
    ncols: int,
    output_filename: str = "compare.png",
):
    blocks = []
    for idx, env_id in enumerate(env_ids):
        blocks += [
            wb.PanelGrid(
                runsets=[runsets[idx].report_runset for runsets in runsetss],
                panels=[
                    wb.LinePlot(
                        x="global_step",
                        y=["charts/episodic_return"],
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
                        y=["charts/episodic_return"],
                        title=env_id,
                        title_y="Episodic Return",
                        max_runs_to_show=100,
                        smoothing_factor=0.8,
                        groupby_rangefunc="stderr",
                        legend_template="${runsetName}",
                    ),
                    # wb.MediaBrowser(
                    #     num_columns=2,
                    #     media_keys="videos",
                    # ),
                ],
            ),
        ]

    nrows = np.ceil(len(env_ids) / ncols).astype(int)
    figsize = (ncols * 4, nrows * 3)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        # sharex=True,
        # sharey=True,
    )

    for idx, env_id in enumerate(env_ids):
        ex = expt.Experiment("Comparison")
        for runsets in runsetss:
            h = create_hypothesis(runsets[idx].name, runsets[idx].runs)
            ex.add_hypothesis(h)
        ax = axes.flatten()[idx]
        ex.plot(
            ax=ax,
            title=env_id,
            x="_runtime",
            y="charts/episodic_return",
            err_style="band",
            std_alpha=0.1,
            rolling=50,
            n_samples=400,
            legend=False,
        )

    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2)
    fig.subplots_adjust(down=0.8)
    # remove the empty axes
    for ax in axes.flatten()[len(env_ids) :]:
        ax.remove()

    print(f"saving figure to {output_filename}")
    plt.savefig(f"{output_filename}", bbox_inches="tight")
    plt.savefig(f"{output_filename.replace('.png', '.pdf')}", bbox_inches="tight")
    return blocks


if __name__ == "__main__":
    env_ids = [
        "HalfCheetah-v2",
        "Hopper-v2",
        "Walker2d-v2",
    ]
    exp_names = [
        "sac_jax",
        "sac_continuous_action",
        "sac_continuous_action_deter_eval",
    ]
    runsetss = []
    for exp_name in exp_names:
        runsetss += [
            [
                Runset(
                    name=f"CleanRL's {exp_name}",
                    filters={"$and": [{"config.env_id.value": env_id}, {"config.exp_name.value": exp_name}]},
                    entity="openrlbenchmark",
                    project="cleanrl",
                    groupby="exp_name",
                )
                for env_id in env_ids
            ]
        ]

    blocks = compare(runsetss, env_ids, output_filename="compare.png", ncols=2)

    print("saving report")
    report = wb.Report(
        project="cleanrl",
        title=f"Compare {exp_names}",
        blocks=blocks,
    )
    report.save()
    print(f"view the generated report at {report.url}")
