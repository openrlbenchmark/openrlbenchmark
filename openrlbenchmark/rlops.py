import argparse
import os
from distutils.util import strtobool
from typing import List
from urllib.parse import parse_qs, urlparse

import expt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import wandb.apis.reports as wb  # noqa
from expt import Hypothesis, Run
from rich.console import Console
from rich.pretty import pprint
import openrlbenchmark
import openrlbenchmark.cache

api = wandb.Api()


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--filters", nargs="+", action="append", default=[],
        help="the filters of the experiments; see docs")
    parser.add_argument("--env-ids", nargs="+", default=["Hopper-v2", "Walker2d-v2", "HalfCheetah-v2"],
        help="the ids of the environment to compare")
    parser.add_argument("--output-filename", type=str, default="compare",
        help="the output filename of the plot, without extension")
    parser.add_argument("--rolling", type=int, default=100,
        help="the rolling window for smoothing the curves")
    parser.add_argument("--metric-last-n-average-window", type=int, default=100,
        help="the last n number of episodes to average metric over in the result table")
    parser.add_argument("--ncols", type=int, default=2,
        help="the number of columns in the chart")
    parser.add_argument("--ncols-legend", type=int, default=2,
        help="the number of legend columns in the chart")
    parser.add_argument("--scan-history", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, we will pull the complete metrics from wandb instead of sampling 500 data points (recommended for generating tables)")
    parser.add_argument("--check-empty-runs", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, we will check for empty wandb runs")
    parser.add_argument("--report", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, a wandb report will be created")
    # fmt: on
    return parser.parse_args()


def create_hypothesis(
    name: str, wandb_runs: List[wandb.apis.public.Run], scan_history: bool = False, metric: str = ""
) -> Hypothesis:
    runs = []
    for idx, run in enumerate(wandb_runs):
        print(run, run.url)
        if scan_history:
            # equivalent to `run_df = pd.DataFrame([row for row in run.scan_history()])`
            run = openrlbenchmark.cache.CachedRun(run, cache_dir=os.path.join(openrlbenchmark.__path__[0], "dataset"))
            run_df = run.run_df
        else:
            run_df = run.history(samples=1500)
        if "videos" in run_df:
            run_df = run_df.drop(columns=["videos"], axis=1)
        if len(metric) > 0:
            run_df["charts/episodic_return"] = run_df[metric]
        runs += [Run(f"seed{idx}", run_df)]
    return Hypothesis(name, runs)


class Runset:
    def __init__(
        self,
        name: str,
        filters: dict,
        entity: str,
        project: str,
        groupby: str = "",
        exp_name: str = "exp_name",
        metric: str = "charts/episodic_return",
        color: str = "#000000",
    ):
        self.name = name
        self.filters = filters
        self.entity = entity
        self.project = project
        self.groupby = groupby
        self.exp_name = exp_name
        self.metric = metric
        self.color = color

    @property
    def runs(self):
        return wandb.Api().runs(path=f"{self.entity}/{self.project}", filters=self.filters)

    @property
    def report_runset(self):
        return wb.Runset(
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
    ncols_legend: int,
    rolling: int,
    metric_last_n_average_window: int,
    scan_history: bool = False,
    output_filename: str = "compare",
    report: bool = False,
):
    blocks = []
    if report:
        for idx, env_id in enumerate(env_ids):
            metric_over_step = wb.LinePlot(
                x="global_step",
                y=list({runsets[idx].metric for runsets in runsetss}),
                title=env_id,
                title_x="Steps",
                title_y="Episodic Return",
                max_runs_to_show=100,
                smoothing_factor=0.8,
                groupby_rangefunc="stderr",
                legend_template="${runsetName}",
            )
            metric_over_step.config["aggregateMetrics"] = True
            metric_over_time = wb.LinePlot(
                x="_runtime",
                y=list({runsets[idx].metric for runsets in runsetss}),
                title=env_id,
                title_y="Episodic Return",
                max_runs_to_show=100,
                smoothing_factor=0.8,
                groupby_rangefunc="stderr",
                legend_template="${runsetName}",
            )
            metric_over_time.config["aggregateMetrics"] = True
            pg = wb.PanelGrid(
                runsets=[runsets[idx].report_runset for runsets in runsetss],
                panels=[
                    metric_over_step,
                    metric_over_time,
                    # wb.MediaBrowser(
                    #     num_columns=2,
                    #     media_keys="videos",
                    # ),
                ],
            )
            custom_run_colors = {}
            for runsets in runsetss:
                custom_run_colors.update(
                    {(runsets[idx].report_runset.name, runsets[idx].runs[0].config[runsets[idx].exp_name]): runsets[idx].color}
                )
            pg.custom_run_colors = custom_run_colors  # IMPORTANT: custom_run_colors is implemented as a custom `setter` that needs to be overwritten unlike regular dictionaries
            blocks += [pg]

    nrows = np.ceil(len(env_ids) / ncols).astype(int)
    figsize = (ncols * 4, nrows * 3)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        # sharex=True,
        # sharey=True,
    )
    if len(env_ids) == 1:
        axes = np.array([axes])
    axes_flatten = axes.flatten()

    result_table = pd.DataFrame(index=env_ids, columns=[runsets[0].name for runsets in runsetss])
    for idx, env_id in enumerate(env_ids):
        ex = expt.Experiment("Comparison")
        for runsets in runsetss:
            h = create_hypothesis(runsets[idx].name, runsets[idx].runs, scan_history, runsets[idx].metric)
            ex.add_hypothesis(h)

        # for each run `i` get the average of the last `rolling` episodes as r_i
        # then take the average and std of r_i as the results.
        result = []
        for hypothesis in ex.hypotheses:
            raw_result = []
            for run in hypothesis.runs:
                raw_result += [run.df["charts/episodic_return"].dropna()[-metric_last_n_average_window:].mean()]
            raw_result = np.array(raw_result)
            result += [f"{raw_result.mean():.2f} ± {raw_result.std():.2f}"]
        result_table.loc[env_id] = result

        ax = axes_flatten[idx]
        ex.plot(
            ax=ax,
            title=env_id,
            x="global_step",
            y="charts/episodic_return",
            err_style="band",
            std_alpha=0.1,
            rolling=rolling,
            colors=[runsets[idx].color for runsets in runsetss],
            # n_samples=500,
            legend=False,
        )

    print(result_table)
    result_table.to_markdown(open(f"{output_filename}.md", "w"))
    result_table.to_csv(open(f"{output_filename}.csv", "w"))

    # add legend
    h, l = axes_flatten[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncols_legend, bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure)

    # remove the empty axes
    for ax in axes_flatten[len(env_ids) :]:
        ax.remove()

    print(f"saving figure to {output_filename}")
    if os.path.dirname(output_filename) != "":
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(f"{output_filename}.png", bbox_inches="tight")
    plt.savefig(f"{output_filename}.pdf", bbox_inches="tight")
    return blocks


if __name__ == "__main__":
    args = parse_args()
    # raise
    console = Console()
    blocks = []
    runsetss = []

    colors_flatten = sns.color_palette(n_colors=sum(len(filters) - 1 for filters in args.filters)).as_hex()
    colors = []
    for filters in args.filters:
        colors += [colors_flatten[: len(filters) - 1]]
        colors_flatten = colors_flatten[len(filters) - 1 :]

    for filters_idx, filters in enumerate(args.filters):
        parse_result = urlparse(filters[0])
        query = parse_qs(parse_result.query)
        metric = query["metric"][0] if "metric" in query else "charts/episodic_return"
        wandb_project_name = query["wpn"][0] if "wpn" in query else args.wandb_project_name
        wandb_entity = query["we"][0] if "we" in query else args.wandb_entity
        custom_env_id_key = query["ceik"][0] if "ceik" in query else "env_id"
        custom_exp_name = query["cen"][0] if "cen" in query else "exp_name"
        pprint(
            {
                "wandb_project_name": wandb_project_name,
                "wandb_entity": wandb_entity,
                "custom_env_id_key": custom_env_id_key,
                "custom_exp_name": custom_exp_name,
                "metric": metric,
            },
            expand_all=True,
        )
        # raise

        for filter_str, color in zip(filters[1:], colors[filters_idx]):
            print("=========", filter_str)
            # parse filter string
            parse_result = urlparse(filter_str)
            exp_name = parse_result.path
            query = parse_qs(parse_result.query)
            user = [{"username": query["user"][0]}] if "user" in query else []
            include_tag_groups = [{"tags": {"$in": [tag]}} for tag in query["tag"]] if "tag" in query else []

            runsets = []
            for env_id in args.env_ids:
                # HACK
                if exp_name == "baselines-ppo2-cnn":
                    env_id = env_id.replace("-v5", "NoFrameskip-v4")
                if exp_name == "ppo_continuous_action" and "rlops-pilot" in query["tag"]:
                    env_id = env_id.replace("-v4", "-v2")

                runsets += [
                    Runset(
                        name=f"{exp_name} ({query})",
                        filters={
                            "$and": [
                                {f"config.{custom_env_id_key}.value": env_id},
                                *include_tag_groups,
                                *user,
                                {f"config.{custom_exp_name}.value": exp_name},
                            ]
                        },
                        entity=wandb_entity,
                        project=wandb_project_name,
                        groupby=custom_exp_name,
                        exp_name=custom_exp_name,
                        metric=metric,
                        color=color,
                    )
                ]
                if args.check_empty_runs:
                    console.print(f"{exp_name} [green]({query})[/] in [purple]{env_id}[/] has {len(runsets[-1].runs)} runs")
                    for run in runsets[-1].runs:
                        console.print(f"┣━━ [link={run.url}]{run.name}[/link] with tags = {run.tags}")
                    assert len(runsets[0].runs) > 0, f"{exp_name} ({query}) in {env_id} has no runs"
            runsetss += [runsets]

    blocks = compare(
        runsetss,
        args.env_ids,
        output_filename=args.output_filename,
        ncols=args.ncols,
        ncols_legend=args.ncols_legend,
        rolling=args.rolling,
        metric_last_n_average_window=args.metric_last_n_average_window,
        scan_history=args.scan_history,
        report=args.report,
    )
    if args.report:
        print("saving report")
        report = wb.Report(
            project="cleanrl",
            title=f"Regression Report: {exp_name}",
            description=str(args.filters),
            blocks=blocks,
        )
        report.save()
        print(f"view the generated report at {report.url}")
