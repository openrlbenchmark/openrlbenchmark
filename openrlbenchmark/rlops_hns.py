import argparse
import copy
import os
from distutils.util import strtobool
import pickle
from typing import List, cast
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
from rich.table import Table

import openrlbenchmark
import openrlbenchmark.cache
from openrlbenchmark.hns import atari_human_normalized_scores as atari_hns

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
    parser.add_argument("--time-unit", type=str, default="m", choices=["s", "m", "h"],
        help="the unit of time in the x-axis of the chart (e.g., `s` for seconds, `m` for minutes, `h` for hours); default: `m`")
    parser.add_argument("--report", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, a wandb report will be created")
    parser.add_argument("--xlabel", type=str, default="Step",
        help="the label of the x-axis")
    parser.add_argument("--ylabel", type=str, default="Human-noramlized Score",
        help="the label of the y-axis")
    # fmt: on
    return parser.parse_args()


def to_rich_table(df: pd.DataFrame) -> Table:
    table = Table()
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    return table


def create_hypothesis(
    name: str, wandb_runs: List[wandb.apis.public.Run], scan_history: bool = False, metric: str = ""
) -> Hypothesis:
    runs = []
    for idx, run in enumerate(wandb_runs):
        print("loading", run, run.url)
        if run.state == "running":
            print(f"Skipping running run: {run}")
            continue
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
        cleaned_df = run_df[["global_step", "_runtime", "charts/episodic_return"]].dropna()
        runs += [Run(f"seed{idx}", cleaned_df)]
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
    console: Console,
    runsetss: List[List[Runset]],
    env_ids: List[str],
    ncols: int,
    ncols_legend: int,
    rolling: int,
    metric_last_n_average_window: int,
    scan_history: bool = False,
    output_filename: str = "compare",
    report: bool = False,
    time_unit: str = "m",
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
    fig_time, axes_time = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        # sharex=True,
        # sharey=True,
    )
    if len(env_ids) == 1:
        axes = np.array([axes])
        axes_time = np.array([axes_time])
    axes_flatten = axes.flatten()
    axes_time_flatten = axes_time.flatten()

    result_table = pd.DataFrame(index=env_ids, columns=[runsets[0].name for runsets in runsetss])
    min_num_seeds_per_hypothesis = {}
    for runsets in runsetss:
        min_num_seeds_per_hypothesis[runsets[0].name] = float("inf")
    exs = []
    runtimes = []
    for idx, env_id in enumerate(env_ids):
        print(f"collecting runs for {env_id}")
        ex = expt.Experiment("Comparison")
        for runsets in runsetss:
            h = create_hypothesis(runsets[idx].name, runsets[idx].runs, scan_history, runsets[idx].metric)
            ex.add_hypothesis(h)
        exs.append(ex)

        # for each run `i` get the average of the last `rolling` episodes as r_i
        # then take the average and std of r_i as the results.
        result = []
        for hypothesis in ex.hypotheses:
            metric_result = []
            console.print(f"{hypothesis.name} has {len(hypothesis.runs)} runs", style="bold")
            min_num_seeds_per_hypothesis[hypothesis.name] = min(min_num_seeds_per_hypothesis[hypothesis.name], len(hypothesis.runs))
            
            for run in hypothesis.runs:
                # calculate hns
                run.df["charts/episodic_return"] = (run.df["charts/episodic_return"] - atari_hns[env_id][0]) / (atari_hns[env_id][1] - atari_hns[env_id][0])
                metric_result += [run.df["charts/episodic_return"].dropna()[-metric_last_n_average_window:].mean()]

                # convert time unit in place
                if time_unit == "m":
                    run.df["_runtime"] /= 60
                elif time_unit == "h":
                    run.df["_runtime"] /= 3600
            metric_result = np.array(metric_result)
            result += [f"{metric_result.mean():.2f} ± {metric_result.std():.2f}"] # , {np.mean(runtimes):.2f} ± {np.std(runtimes):.2f}
        result_table.loc[env_id] = result
        runtimes.append(list(ex.summary()["_runtime"]))
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
            legend=False,
        )
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)
        ax_time = axes_time_flatten[idx]
        ex.plot(
            ax=ax_time,
            title=env_id,
            x="_runtime",
            y="charts/episodic_return",
            err_style="band",
            std_alpha=0.1,
            rolling=rolling,
            colors=[runsets[idx].color for runsets in runsetss],
            legend=False,
        )
        ax_time.set_ylabel(args.ylabel)
        ax_time.set_xlabel(f"Time ({time_unit})")
    runtimes = pd.DataFrame(np.array(runtimes), index=env_ids, columns=list(ex.summary()["name"]))
    console.rule(f"[bold red]Runtime ({time_unit}) (mean ± std)")
    console.print(to_rich_table(runtimes.rename_axis("Environment").reset_index()))

    # for each run set, for each seed, plot 57 curves and get their median curves, then plot the average of the median curves
    hns_ex = expt.Experiment("Human Normalized Score")
    for runsets_idx, runsets in enumerate(runsetss):
        # for each seed
        median_curves = []
        for seed_idx, _ in enumerate(range(min_num_seeds_per_hypothesis[runsets[0].name])): # exs[0][runsets_idx]
            print(f"collecting runs for {runsets[0].name} seed {seed_idx}")

            runs_of_one_seed = []
            for ex_idx, ex in enumerate(exs):
                runs_of_one_seed.append(ex[runsets_idx][seed_idx])
            temp_fig, temp_axe = plt.subplots(nrows=1, ncols=1)
            Hypothesis("seed", runs_of_one_seed).plot(
                ax=temp_axe,
                title="Human Normalized Score",
                x="global_step",
                y="charts/episodic_return",
                err_style="band",
                rolling=rolling,
                representative_fn=lambda h: cast(pd.DataFrame, h.grouped.median()), # median curve
                legend=False,
            )
            # Hypothesis("seed", runs_of_one_seed).plot( ax=temp_axe, title="Human Normalized Score", x="global_step", y="charts/episodic_return", err_style="band", rolling=rolling, representative_fn=lambda h: cast(pd.DataFrame, h.grouped.median()))
            assert len(temp_axe.lines) == 1
            median_of_runs_of_one_seed = temp_axe.lines[0].get_xydata()
            median_curves.append(median_of_runs_of_one_seed)
            plt.close(temp_fig)
        
        runs = []
        for median_curve_idx, median_curve in enumerate(median_curves):
            run_df = pd.DataFrame(median_curve, columns=["global_step", "charts/episodic_return"])
            runs.append(Run(f"median_curve_idx:{median_curve_idx}", run_df))
        hns_ex.add_hypothesis(Hypothesis(runsets[0].name, runs))

    ncols = 2
    nrows = 1
    figsize = (ncols * 4, nrows * 3)
    fig_median_hns, axes_median_hns = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharey=True,
    )
    hns_ex.plot(
        title=" ",
        ax=axes_median_hns[0],
        x="global_step",
        y="charts/episodic_return",
        err_style="band",
        std_alpha=0.1,
        rolling=rolling,
        colors=[runsets[0].color for runsets in runsetss],
        legend=False,
    )
    hns_ex_time = copy.deepcopy(hns_ex)
    for hypo_idx, hypo in enumerate(hns_ex_time.hypotheses):
        for run_idx in range(len(hypo.runs)):
            hypo.runs[run_idx].df["_runtime"] = np.linspace(0, runtimes.mean(axis=0).iloc[hypo_idx], len(hypo.runs[run_idx].df))

    axes_median_hns[0].set_ylabel(args.ylabel)
    axes_median_hns[0].set_xlabel(f"Steps")
    hns_ex_time.plot(
        title=" ",
        ax=axes_median_hns[1],
        x="_runtime",
        y="charts/episodic_return",
        err_style="band",
        std_alpha=0.1,
        rolling=rolling,
        colors=[runsets[0].color for runsets in runsetss],
        legend=False,
    )
    h, l = axes_median_hns[0].get_legend_handles_labels()
    fig_median_hns.legend(h, l, loc="lower center", ncol=ncols_legend, bbox_to_anchor=(0.5, 0.9), bbox_transform=fig_median_hns.transFigure)
    fig_median_hns.savefig(f"{output_filename}_median_hns.png", bbox_inches="tight")
    fig_median_hns.savefig(f"{output_filename}_median_hns.pdf", bbox_inches="tight")


    # with open(f"{output_filename}_median_hns.pkl", 'wb') as f:
    #     pickle.dump([line.get_xydata() for line in ax_median_hns.lines], f)
    console.rule(f"[bold red]{args.ylabel} (mean ± std)")
    console.print(to_rich_table(result_table.rename_axis("Environment").reset_index()))
    result_table.to_markdown(open(f"{output_filename}.md", "w"))
    result_table.to_csv(open(f"{output_filename}.csv", "w"))
    runtimes.to_markdown(open(f"{output_filename}_runtimes.md", "w"))
    runtimes.to_csv(open(f"{output_filename}_runtimes.csv", "w"))
    console.rule(f"[bold red]Runtime ({time_unit}) Average")
    average_runtime = pd.DataFrame(runtimes.mean(axis=0)).reset_index()
    average_runtime.columns = ["Environment", "Average Runtime"]
    console.print(to_rich_table(average_runtime))

    # add legend
    h, l = axes_flatten[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncols_legend, bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure)
    h, l = axes_time_flatten[0].get_legend_handles_labels()
    fig_time.legend(
        h, l, loc="lower center", ncol=ncols_legend, bbox_to_anchor=(0.5, 1.0), bbox_transform=fig_time.transFigure
    )

    # remove the empty axes
    for ax in axes_flatten[len(env_ids) :]:
        ax.remove()
    for ax in axes_time_flatten[len(env_ids) :]:
        ax.remove()

    print(f"saving figures and tables to {output_filename}")
    if os.path.dirname(output_filename) != "":
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fig.savefig(f"{output_filename}.png", bbox_inches="tight")
    fig.savefig(f"{output_filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_filename}.svg", bbox_inches="tight")
    fig_time.savefig(f"{output_filename}-time.png", bbox_inches="tight")
    fig_time.savefig(f"{output_filename}-time.pdf", bbox_inches="tight")
    fig_time.savefig(f"{output_filename}-time.svg", bbox_inches="tight")
    return blocks


if __name__ == "__main__":
    args = parse_args()
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

        for filter_str, color in zip(filters[1:], colors[filters_idx]):
            print("=========", filter_str)
            # parse filter string
            parse_result = urlparse(filter_str)
            exp_name = parse_result.path
            query = parse_qs(parse_result.query)
            user = [{"username": query["user"][0]}] if "user" in query else []
            include_tag_groups = [{"tags": {"$in": [tag]}} for tag in query["tag"]] if "tag" in query else []
            custom_legend = query["cl"][0] if "cl" in query else ""

            # HACK unescape
            custom_legend = custom_legend.replace("\\n", "\n")

            runsets = []
            for env_id in args.env_ids:
                # HACK
                if "alepy" in exp_name:  # alepy experiments: `Breakout-v5` -> `ALE/Breakout-v5`
                    env_id = f"ALE/{env_id}"
                elif "envpool" not in exp_name:
                    env_id = env_id.replace("-v4", "-v2")  # mujoco experiments: `HalfCheetah-v4` -> `HalfCheetah-v2`
                    env_id = env_id.replace(
                        "-v5", "NoFrameskip-v4"
                    )  # old atari experiments: `Breakout-v5` -> `BreakoutNoFrameskip-v4`
                if exp_name == "ppo_continuous_action" and "rlops-pilot" in query["tag"]:
                    env_id = env_id.replace("-v4", "-v2")

                runsets.append(
                    Runset(
                        name=f"{wandb_entity}/{wandb_project_name}/{exp_name} ({query})"
                        if custom_legend == ""
                        else custom_legend,
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
                )
                if args.check_empty_runs:
                    console.print(f"{exp_name} [green]({query})[/] in [purple]{env_id}[/] has {len(runsets[-1].runs)} runs")
                    for run in runsets[-1].runs:
                        console.print(f"┣━━ [link={run.url}]{run.name}[/link] with tags = {run.tags}")
                    assert len(runsets[0].runs) > 0, f"{exp_name} ({query}) in {env_id} has no runs"
            runsetss.append(runsets)

    blocks = compare(
        console,
        runsetss,
        args.env_ids,
        output_filename=args.output_filename,
        ncols=args.ncols,
        ncols_legend=args.ncols_legend,
        rolling=args.rolling,
        metric_last_n_average_window=args.metric_last_n_average_window,
        scan_history=args.scan_history,
        report=args.report,
        time_unit=args.time_unit,
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
