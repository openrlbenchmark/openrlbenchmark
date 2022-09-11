from typing import List
import expt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import wandb.apis.reports as wb  # noqa
from expt import Run
from expt.plot import GridPlot

from openrlbenchmark.atari_data import atari_human_normalized_scores
from openrlbenchmark import Runset, create_expt_runs
wandb.require("report-editing")

def plot_atari(runsets: List[Runset], return_wandb_report_blocks=False):
    env_ids = atari_human_normalized_scores.keys()
    hms = []
    raw_scores = []
    NUM_FRAME_STACK = 4
    NUM_COLS = 5
    g = GridPlot(y_names=env_ids, layout=(int(np.ceil(len(env_ids) / NUM_COLS)), NUM_COLS))
    blocks = []




    for env_id in env_ids:
        if return_wandb_report_blocks:
            l1 = wb.LinePlot(
                x="global_step",
                y=[runset.y_axis for runset in runsets],
                title=env_id,
                title_x="Steps",
                title_y="Episodic Return",
                max_runs_to_show=100,
                smoothing_factor=0.8,
                groupby_rangefunc="stderr",
                legend_template="${runsetName}",
            )
            l2 = wb.LinePlot(
                x="_runtime",
                y=[runset.y_axis for runset in runsets],
                title=env_id,
                title_y="Episodic Return",
                max_runs_to_show=100,
                smoothing_factor=0.8,
                groupby_rangefunc="stderr",
                legend_template="${runsetName}",
            )
            l1.config["aggregateMetrics"] = True
            l2.config["aggregateMetrics"] = True
            blocks += [
                wb.PanelGrid(
                    runsets=[runset.report_runset(env_id=env_id) for runset in runsets],
                    panels=[
                        l1,
                        l2,
                    ],
                ),
            ]

        dummy_y_axis = runsets[0].y_axis
        ex = expt.Experiment("Comparison of PPO")
        for runset in runsets:
            wandb_runs = runset.runs(env_id=env_id)
            expt_runs = create_expt_runs(wandb_runs)

            # normalize scores and adjust x-axis from steps to frames
            for expt_run in expt_runs:
                expt_run.df["human normalized score"] = (
                    expt_run.df[runset.y_axis] - atari_human_normalized_scores[env_id][0]
                ) / (atari_human_normalized_scores[env_id][1] - atari_human_normalized_scores[env_id][0])
                expt_run.df["global_step"] *= NUM_FRAME_STACK
                expt_run.df[dummy_y_axis] = expt_run.df[runset.y_axis]

            if len(wandb_runs) > 0:
                ex.add_runs(runset.name, expt_runs)
        ex.plot(
            ax=g[env_id],
            title=env_id,
            x="_runtime",
            y=dummy_y_axis,
            err_style="band",
            std_alpha=0.1,
            rolling=50,
            n_samples=400,
            legend=False,
        )
        ax = g[env_id]
        ax2 = ax.twinx()
        ax2.set_ylim([0, max(ex.summary()["human normalized score"])])
        raw_scores += [list(ex.summary()[dummy_y_axis])]
        hms += [list(ex.summary()["human normalized score"])]

    g.add_legend(ax=g.axes[-1, -1], loc="upper left", bbox_to_anchor=(0, 1))
    for ax in g.axes_active:
        ax.xaxis.set_label_text("")
        ax.yaxis.set_label_text("")

    plt.tight_layout(w_pad=1)
    plt.savefig("static/hms_each_game.png")
    plt.savefig("static/hms_each_game.svg")

    pd.DataFrame(sorted(zip(env_ids, *[np.array(raw_scores)[:,i] for i in range(len(raw_scores[0]))])), columns=["Environment"] + [runset.name for runset in runsets]).set_index("Environment").to_markdown(
        "static/atari_returns.md",
    )
    pd.DataFrame(sorted(zip(env_ids, *[np.array(hms)[:,i] for i in range(len(hms[0]))])), columns=["Environment"] + [runset.name for runset in runsets]).set_index("Environment").to_markdown(
        "static/atari_hns.md",
    )
    plt.clf()
    plt.rcdefaults()
    for i in range(len(hms[0])):
        sorted_tuple = sorted(zip(np.array(hms)[:,i], env_ids))
        sorted_hms = [x for x, _ in sorted_tuple]
        sorted_env_ids = [x for _, x in sorted_tuple]
        fig, ax = plt.subplots(figsize=(7, 10))
        y_pos = np.arange(len(sorted_env_ids))
        bars = ax.barh(y_pos, sorted_hms)
        ax.bar_label(bars, fmt="%.2f")
        ax.set_yticks(y_pos, labels=sorted_env_ids)
        plt.title(f"{runsets[i].name} Human Normalized Scores")
        plt.tight_layout()
        plt.savefig(f"static/runset_{i}_hms_bar.png")
        plt.savefig(f"static/runset_{i}_hms_bar.svg")

    return blocks
    if return_wandb_report_blocks:
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
