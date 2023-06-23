import copy
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
from urllib.parse import parse_qs, urlparse

import expt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peewee as pw
import seaborn as sns
import tyro
import wandb
import wandb.apis.reports as wb  # noqa
from dotmap import DotMap
from expt import Hypothesis, Run
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from rliable import library as rly
from rliable import metrics, plot_utils

import openrlbenchmark
import openrlbenchmark.cache
from openrlbenchmark.hns import atari_human_normalized_scores as atari_hns
from openrlbenchmark.offline_db import OfflineRun, OfflineRunTag, Tag, database_proxy


@dataclass
class RliableConfig:
    nsubsamples: int = 20
    """the number of subsamples for rliable"""
    score_normalization_method: Literal["maxmin", "atari"] = "maxmin"
    """the method to normalize the scores"""
    normalized_score_threshold: float = 8.0
    """the threshold for the normalized score for the performance profile"""
    sample_efficiency_plots: bool = True
    """if toggled, we will generate sample efficiency plots"""
    sample_efficiency_and_walltime_efficiency_method: Optional[Literal["Median", "IQM", "Mean", "Optimality Gap"]] = "Median"
    """the method to compute the sample efficiency and walltime efficiency"""
    performance_profile_plots: bool = True
    """if toggled, we will generate performance profile plots"""
    aggregate_metrics_plots: bool = True
    """if toggled, we will generate aggregate metrics plots"""
    sample_efficiency_num_bootstrap_reps: int = 10  # 50000
    """the number of bootstrap replications in `rliable` to use for computing the sample efficiency"""
    performance_profile_num_bootstrap_reps: int = 10  # 2000
    """the number of bootstrap replications in `rliable` to use for computing the performance profile"""
    interval_estimates_num_bootstrap_reps: int = 10  # 2000
    """the number of bootstrap replications in `rliable` to use for computing the the interval estimates"""


@dataclass
class PlotConfig:
    ncols: int = 2
    """the number of columns in the chart"""
    nrows: tyro.conf.Suppress[int] = None
    """(TO BE FILLED in runtime) the number of rows in the chart"""
    ncols_legend: int = 2
    """the number of legend columns in the chart"""
    xlabel: str = "Steps"
    """the label of the x-axis"""
    ylabel: str = "Episodic Return"
    """the label of the y-axis"""
    sharex: bool = False
    """if toggled, we will share the x-axis across all subplots"""
    max_steps: int = None
    """if specified, the maximum number of steps to plot"""
    rolling: int = 100
    """the rolling window for smoothing the curves"""
    time_unit: str = "m"
    """the unit of time in the x-axis of the chart (e.g., `s` for seconds, `m` for minutes, `h` for hours); default: `m`"""
    cm: float = 4.0
    """the multiplier for the column width"""
    rm: float = 3.0
    """the multiplier for the row height"""
    hspace: float = None
    """the height space between subplots"""
    wspace: float = None
    """the width space between subplots"""


@dataclass
class Args:
    filters: tyro.conf.UseAppendAction[List[List[str]]]
    """the filters of the experiments; see docs"""
    env_ids: tyro.conf.UseAppendAction[List[List[str]]]
    """the ids of the environment to compare"""
    output_filename: str = "compare"
    """the output filename of the plot, without extension"""
    metric_last_n_average_window: int = 100
    """the last n number of episodes to average metric over in the result table"""
    scan_history: bool = False
    """if toggled, we will pull the complete metrics from wandb instead of sampling 500 data points (recommended for generating tables)"""
    check_empty_runs: bool = True
    """if toggled, we will check for empty wandb runs"""
    report: bool = False
    """if toggled, a wandb report will be created"""
    offline: bool = False
    """if toggled, we will use the offline database instead of wandb"""
    pc: PlotConfig = field(default_factory=PlotConfig)
    """the plot configuration"""
    rliable: bool = False
    """if toggled, we will use rliable to compute the metrics"""
    rc: RliableConfig = field(default_factory=RliableConfig)
    """the rliable configuration"""


class Runset:
    def __init__(
        self,
        name: str,
        entity: str,
        project: str,
        metrics: List[str] = ["charts/episodic_return"],
        groupby: str = "",
        custom_exp_name_key: str = "exp_name",
        custom_xaxis_key: str = "global_step",
        exp_name: str = "",
        custom_env_id_key: str = "env_id",
        env_id: str = "",
        tags: List[str] = [],
        username: str = "",
        color: str = "#000000",
        offline_db: pw.Database = None,
        offline: bool = False,
    ):
        self.name = name
        self.entity = entity
        self.project = project
        self.metrics = metrics
        self.groupby = groupby
        self.custom_exp_name_key = custom_exp_name_key
        self.custom_xaxis_key = custom_xaxis_key
        self.exp_name = exp_name
        self.custom_env_id_key = custom_env_id_key
        self.env_id = env_id
        self.color = color
        self.tags = tags
        self.username = username
        self.offline_db = offline_db
        self.offline = offline

        user = [{"username": self.username}] if self.username else []
        include_tag_groups = [{"tags": {"$in": [tag]}} for tag in self.tags] if len(self.tags) > 0 else []
        
        # hack to deal with wandb's nested config
        # click the "View Raw Data" button of the config in
        # https://wandb.ai/costa-huang/cleanRL/runs/3nhnaboz/overview
        # to see how .value is added to the config
        # it should look like this:
        # {
        #     ...
        #     "env_id": { "desc": null, "value": "Pendulum-v1" },
        # }
        # so the correct key is `config.env_id.value`
        # but sometimes configs are stored in a weird way like
        # https://wandb.ai/costa-huang/trl/runs/lpwu2w4g/overview
        # {
        #   "trl_ppo_trainer_config": {
        #     "desc": null,
        #     "value": {
        #       "lam": 0.95,
        #       ...
        #     }
        #   }
        # }
        # so the correct key is `config.trl_ppo_trainer_config.value.lam`
        if ".value" not in self.custom_env_id_key:
            self.custom_env_id_key += ".value"
        if ".value" not in self.custom_exp_name_key:
            self.custom_exp_name_key += ".value"
        self.wandb_filters = {
            "$and": [
                {f"config.{self.custom_env_id_key}": self.env_id},
                *include_tag_groups,
                *user,
                {f"config.{self.custom_exp_name_key}": self.exp_name},
            ]
        }

    @property
    def runs(self):
        if not self.offline:
            return wandb.Api().runs(
                path=f"{self.entity}/{self.project}",
                filters=self.wandb_filters,
            )
        else:
            with self.offline_db.bind_ctx([OfflineRun, OfflineRunTag, Tag]):
                cond = (
                    (OfflineRun.project == self.project)
                    & (OfflineRun.entity == self.entity)
                    & (OfflineRun.config[self.custom_env_id_key] == self.env_id)
                    & (OfflineRun.config[self.custom_exp_name_key] == self.exp_name)
                )
                if self.username:
                    cond = cond and OfflineRun.username == self.username
                if len(self.tags) > 0:
                    for tag_str in self.tags:
                        cond = cond & (Tag.name == tag_str)
                    query = OfflineRun.select().join(OfflineRunTag).join(Tag).where(cond)
                else:
                    query = OfflineRun.select().where(cond)
                g = [run for run in query]

            return g

    @property
    def report_runset(self):
        return wb.Runset(
            name=self.name,
            entity=self.entity,
            project=self.project,
            filters={"$or": [self.wandb_filters]},
            groupby=[self.groupby] if len(self.groupby) > 0 else None,
        )


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table()
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


def create_hypothesis(runset: Runset, target_metrics, scan_history: bool = False) -> Hypothesis:
    runs = []
    for idx, run in enumerate(runset.runs):
        print("loading", run, run.url)
        if run.state == "running":
            print(f"Skipping running run: {run}")
            continue
        if scan_history:
            run = openrlbenchmark.cache.CachedRun(run, cache_dir=os.path.join(openrlbenchmark.__path__[0], "dataset"))
            with runset.offline_db.bind_ctx([OfflineRun, OfflineRunTag, Tag]):
                tags = []
                for tag_str in run.run.tags:
                    tag = Tag.get_or_none(name=tag_str)
                    if not tag:
                        tag = Tag.create(name=tag_str)
                        tag.save()
                    tags.append(tag)
                offline_run = OfflineRun.get_or_none(id=run.run.id)
                if offline_run:
                    offline_run.delete_instance()
                offline_run = OfflineRun.create(
                    id=run.run.id,
                    name=run.run.name,
                    state=run.run.state,
                    url=run.run.url,
                    path=run.run.path,
                    username=run.run.user.username,
                    tags=tags,
                    project=run.run.project,
                    entity=run.run.entity,
                    config=run.run.config.toDict() if isinstance(run.run.config, DotMap) else run.run.config,
                )
                offline_run.save()
            run_df = run.run_df
        else:
            run_df = run.history(samples=1500)
        if "videos" in run_df:
            run_df = run_df.drop(columns=["videos"], axis=1)
        if runset.custom_xaxis_key in run_df:
            run_df["global_step"] = run_df[runset.custom_xaxis_key]
        for source_metric, target_metric in zip(runset.metrics, target_metrics):
            if source_metric in run_df:
                run_df[target_metric] = run_df[source_metric]
        cleaned_df = run_df[["global_step", "_runtime"] + target_metrics].dropna(how="all")
        runs += [Run(f"seed{idx}", cleaned_df)]
    return Hypothesis(runset.name, runs)


def compare(
    console: Console,
    runsetss: List[List[Runset]],
    env_ids: List[str],
    metric_last_n_average_window: int,
    scan_history: bool = False,
    output_filename: str = "compare",
    report: bool = False,
    pc: PlotConfig = None,
):
    blocks = []
    if report:
        for idx, env_id in enumerate(env_ids):
            metrics_over_step = []
            metrics_over_time = []
            for i in range(len(runsetss[0][idx].metrics)):
                metric_over_step = wb.LinePlot(
                    x=runsetss[0][idx].custom_xaxis_key,
                    y=list({runsets[idx].metrics[i] for runsets in runsetss}),
                    title=runsetss[0][idx].metrics[i] + " " + env_id,
                    title_x="Steps",
                    title_y="Episodic Return",
                    max_runs_to_show=100,
                    smoothing_factor=0.8,
                    groupby_rangefunc="stderr",
                    legend_template="${runsetName}",
                )
                metric_over_step.config["aggregateMetrics"] = True
                metrics_over_step.append(metric_over_step)
                metric_over_time = wb.LinePlot(
                    x="_runtime",
                    y=list({runsets[idx].metrics[i] for runsets in runsetss}),
                    title=runsetss[0][idx].metrics[i] + " " + env_id,
                    title_y="Episodic Return",
                    max_runs_to_show=100,
                    smoothing_factor=0.8,
                    groupby_rangefunc="stderr",
                    legend_template="${runsetName}",
                )
                metric_over_time.config["aggregateMetrics"] = True
                metrics_over_time.append(metric_over_time)

            flattened_metrics = [metrics_over_step, metrics_over_time]
            flattened_metrics = [item for sublist in flattened_metrics for item in sublist]
            pg = wb.PanelGrid(
                runsets=[runsets[idx].report_runset for runsets in runsetss],
                panels=flattened_metrics,
            )
            custom_run_colors = {}
            # TODO: color stuff doesn't work because of the filter syntax
            # for runsets in runsetss:
            #     custom_run_colors.update(
            #         {
            #             (
            #                 runsets[idx].report_runset.name,
            #                 runsets[idx].runs[0].config[runsets[idx].custom_exp_name_key],
            #             ): runsets[idx].color
            #         }
            #     )
            pg.custom_run_colors = custom_run_colors  # IMPORTANT: custom_run_colors is implemented as a custom `setter` that needs to be overwritten unlike regular dictionaries
            blocks += [pg]

    figsize = (pc.ncols * pc.cm, pc.nrows * pc.rm)
    fig, axes = plt.subplots(
        nrows=pc.nrows,
        ncols=pc.ncols,
        figsize=figsize,
        sharex=pc.sharex,
        # sharey=True,
    )
    fig_time, axes_time = plt.subplots(
        nrows=pc.nrows,
        ncols=pc.ncols,
        figsize=figsize,
        sharex=pc.sharex,
        # sharey=True,
    )
    if len(env_ids) == 1:
        axes = np.array([axes])
        axes_time = np.array([axes_time])
    axes_flatten = axes.flatten()
    axes_time_flatten = axes_time.flatten()

    result_table = pd.DataFrame(index=env_ids, columns=[runsets[0].name for runsets in runsetss])
    exs = []
    runtimes = []
    global_steps = []
    for idx, env_id in enumerate(env_ids):
        print(f"collecting runs for {env_id}")
        hypotheses = [create_hypothesis(runsets[idx], runsetss[0][idx].metrics, scan_history) for runsets in runsetss]
        ex = expt.Experiment("Comparison", hypotheses)
        exs.append(ex)

        # for each run `i` get the average of the last `rolling` episodes as r_i
        # then take the average and std of r_i as the results.
        result = []
        for hypothesis in ex.hypotheses:
            metric_result = []
            console.print(f"{hypothesis.name} has {len(hypothesis.runs)} runs", style="bold")
            for run in hypothesis.runs:
                # metric_result += [run.df["charts/episodic_return"].dropna()[-metric_last_n_average_window:].mean()]

                # convert time unit in place
                if pc.time_unit == "m":
                    run.df["_runtime"] /= 60
                elif pc.time_unit == "h":
                    run.df["_runtime"] /= 3600
            metric_result = np.array(metric_result)
            result += [f"{metric_result.mean():.2f} ± {metric_result.std():.2f}"]
        result_table.loc[env_id] = result
        runtimes.append(list(ex.summary()["_runtime"]))
        global_steps.append(list(ex.summary()["global_step"]))

        for idx_metric, metric in enumerate(runsetss[0][0].metrics):
            metric_str = metric.replace("eval/", "")
            ax = axes_flatten[len(env_ids) * idx_metric + idx]
            ex.plot(
                ax=ax,
                title=env_id,
                x="global_step",
                y=metric,
                err_style="unit_traces",
                std_alpha=0.1,
                n_samples=10000,
                rolling=pc.rolling,
                colors=[runsets[idx].color for runsets in runsetss],
                legend=False,
            )
            # breakpoint()
            ax.set_xlabel("")
            if idx_metric == 0:
                ax.set_title(env_id)
            else:
                ax.set_title("")
            if idx == 0:
                ax.set_ylabel(metric_str)
            else:
                ax.set_ylabel("")
            if pc.max_steps is not None:
                ax.set_xlim(0, pc.max_steps)
            ax_time = axes_time_flatten[len(env_ids) * idx_metric + idx]
            ex.plot(
                ax=ax_time,
                title=env_id,
                x="_runtime",
                y=metric,
                err_style="band",
                std_alpha=0.1,
                n_samples=10000,
                rolling=pc.rolling,
                colors=[runsets[idx].color for runsets in runsetss],
                legend=False,
            )
            ax_time.set_xlabel("")
            if idx_metric == 0:
                ax_time.set_title(env_id)
            else:
                ax_time.set_title("")
            if idx == 0:
                ax_time.set_ylabel(metric_str)
            else:
                ax_time.set_ylabel("")
    runtimes = pd.DataFrame(np.array(runtimes), index=env_ids, columns=list(ex.summary()["name"]))
    global_steps = pd.DataFrame(np.array(global_steps), index=env_ids, columns=list(ex.summary()["name"]))
    print_rich_table(f"Runtime ({pc.time_unit}) (mean ± std)", runtimes.rename_axis("Environment").reset_index(), console)

    # create the required directory for `output_filename`
    if len(os.path.dirname(output_filename)) > 0:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    print_rich_table(f"{pc.ylabel} (mean ± std)", result_table.rename_axis("Environment").reset_index(), console)
    result_table.to_markdown(open(f"{output_filename}.md", "w"))
    result_table.to_csv(open(f"{output_filename}.csv", "w"))
    runtimes.to_markdown(open(f"{output_filename}_runtimes.md", "w"))
    runtimes.to_csv(open(f"{output_filename}_runtimes.csv", "w"))
    average_runtime = pd.DataFrame(runtimes.mean(axis=0)).reset_index()
    average_runtime.columns = ["Environment", "Average Runtime"]
    print_rich_table(f"Runtime ({pc.time_unit}) Average", average_runtime, console)

    # add legend
    h, l = axes_flatten[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=pc.ncols_legend, bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure)
    fig.supxlabel(pc.xlabel)
    fig.tight_layout()
    h, l = axes_time_flatten[0].get_legend_handles_labels()
    fig_time.legend(
        h, l, loc="lower center", ncol=pc.ncols_legend, bbox_to_anchor=(0.5, 1.0), bbox_transform=fig_time.transFigure
    )
    fig_time.supxlabel(f"Time ({pc.time_unit})")
    fig_time.tight_layout()

    # remove the empty axes
    for ax in axes_flatten[len(env_ids) * len(metrics):]:
        ax.remove()
    for ax in axes_time_flatten[len(env_ids) * len(metrics):]:
        ax.remove()

    print(f"saving figures and tables to {output_filename}")
    fig.subplots_adjust(hspace=pc.hspace, wspace=pc.wspace)
    fig_time.subplots_adjust(hspace=pc.hspace, wspace=pc.wspace)
    fig.savefig(f"{output_filename}.png", bbox_inches="tight")
    fig.savefig(f"{output_filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_filename}.svg", bbox_inches="tight")
    fig_time.savefig(f"{output_filename}-time.png", bbox_inches="tight")
    fig_time.savefig(f"{output_filename}-time.pdf", bbox_inches="tight")
    fig_time.savefig(f"{output_filename}-time.svg", bbox_inches="tight")
    return blocks, runtimes, global_steps, exs


def normalize_score(score_dict: Dict[str, np.ndarray], max_scores: np.ndarray, min_scores: np.ndarray):
    """
    Each item in `score_dict` has shape (num_seeds, num_envs, num_subsamples)
    `max_scores` has shape (num_envs)
    `min_scores` has shape (num_envs)
    """
    normalized_score_dict = {}
    for key in score_dict:
        normalized_score_dict[key] = (score_dict[key] - min_scores.reshape(1, -1, 1)) / (
            max_scores.reshape(1, -1, 1) - min_scores.reshape(1, -1, 1)
        )
    return normalized_score_dict


def maxmin_normalize_score(score_dict: Dict[str, np.ndarray]):
    all_scores = np.concatenate([score_dict[key] for key in score_dict], axis=0)
    max_scores = all_scores.max(0).max(1)  # 1) max over all experiments and seds 2) max over all steps
    min_scores = all_scores.min(0).min(1)  # 1) min over all experiments and seds 2) min over all steps
    return normalize_score(score_dict, max_scores, min_scores)


def atari_normalize_score(score_dict, original_env_ids):
    env_ids = []
    for env_id in original_env_ids:
        if env_id.endswith("NoFrameskip-v4"):
            env_id = env_id.replace("NoFrameskip-v4", "-v5")
        env_ids.append(env_id)
    max_scores = np.array([atari_hns[env_id][1] for env_id in env_ids])
    min_scores = np.array([atari_hns[env_id][0] for env_id in env_ids])
    return normalize_score(score_dict, max_scores, min_scores)


if __name__ == "__main__":
    args = tyro.cli(Args)
    # by default assume all the env_ids are the same
    if len(args.filters) > 1 and len(args.env_ids) == 1:
        args.env_ids = args.env_ids * len(args.filters)

    console = Console()
    blocks = []
    runsetss = []
    offline_dbs = {}
    colors_flatten_original = [c for item in ["deep", "dark", "bright"] for c in sns.color_palette(item).as_hex()]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors_flatten_original)
    colors_flatten = copy.deepcopy(colors_flatten_original)
    colors = []
    for filters in args.filters:
        colors += [colors_flatten[: len(filters) - 1]]
        colors_flatten = colors_flatten[len(filters) - 1 :]

    for filters_idx, filters in enumerate(args.filters):
        parse_result = urlparse(filters[0])
        query = parse_qs(parse_result.query)
        metrics = query["metrics"] if "metrics" in query else ["charts/episodic_return"]
        # calculate the number of rows
        args.pc.nrows = np.ceil(len(args.env_ids[0]) * len(metrics) / args.pc.ncols).astype(int)
        wandb_project_name = query["wpn"][0] if "wpn" in query else args.wandb_project_name
        wandb_entity = query["we"][0] if "we" in query else args.wandb_entity
        custom_env_id_key = query["ceik"][0] if "ceik" in query else "env_id"
        custom_exp_name_key = query["cen"][0] if "cen" in query else "exp_name"
        custom_xaxis_key = query["xaxis"][0] if "xaxis" in query else "global_step"
        pprint(
            {
                "wandb_project_name": wandb_project_name,
                "wandb_entity": wandb_entity,
                "custom_env_id_key": custom_env_id_key,
                "custom_exp_name_key": custom_exp_name_key,
                "custom_xaxis_key": custom_xaxis_key,
                "metrics": metrics,
            },
            expand_all=True,
        )
        if f"{wandb_entity}/{wandb_project_name}" not in offline_dbs:
            offline_db_folder = os.path.join(openrlbenchmark.__path__[0], "dataset", f"{wandb_entity}/{wandb_project_name}")
            offline_db_path = os.path.join(offline_db_folder, "offline.sqlite")
            print(offline_db_path)
            os.makedirs(offline_db_folder, exist_ok=True)
            offline_db = pw.SqliteDatabase(offline_db_path)
            database_proxy.initialize(offline_db)
            offline_db.connect()
            offline_db.create_tables([OfflineRun, Tag, OfflineRunTag])
            offline_dbs[f"{wandb_entity}/{wandb_project_name}"] = offline_db

        for filter_str, color in zip(filters[1:], colors[filters_idx]):
            print("=========", filter_str)
            # parse filter string
            parse_result = urlparse(filter_str)
            exp_name = parse_result.path
            query = parse_qs(parse_result.query)
            username = query["user"][0] if "user" in query else None
            tags = query["tag"] if "tag" in query else []
            custom_legend = query["cl"][0] if "cl" in query else ""
            # HACK unescape
            custom_legend = custom_legend.replace("\\n", "\n")

            runsets = []
            for env_id in args.env_ids[filters_idx]:
                runsets.append(
                    Runset(
                        name=f"{wandb_entity}/{wandb_project_name}/{exp_name} ({query})"
                        if custom_legend == ""
                        else custom_legend,
                        entity=wandb_entity,
                        project=wandb_project_name,
                        metrics=metrics,
                        groupby=custom_exp_name_key,
                        custom_exp_name_key=custom_exp_name_key,
                        custom_xaxis_key=custom_xaxis_key,
                        exp_name=exp_name,
                        custom_env_id_key=custom_env_id_key,
                        env_id=env_id,
                        tags=tags,
                        username=username,
                        color=color,
                        offline_db=offline_dbs[f"{wandb_entity}/{wandb_project_name}"],
                        offline=args.offline,
                    )
                )
                if args.check_empty_runs:
                    console.print(f"{exp_name} [green]({query})[/] in [purple]{env_id}[/] has {len(runsets[-1].runs)} runs")
                    for run in runsets[-1].runs:
                        console.print(f"┣━━ [link={run.url}]{run.name}[/link] with tags = {run.tags}")
                    print(runsets[0].wandb_filters)
                    assert len(runsets[0].runs) > 0, f"{exp_name} ({query}) in {env_id} has no runs"
            runsetss.append(runsets)

    blocks, runtimes, global_steps, exs = compare(
        console,
        runsetss,
        args.env_ids[0],
        output_filename=args.output_filename,
        metric_last_n_average_window=args.metric_last_n_average_window,
        scan_history=args.scan_history,
        report=args.report,
        pc=args.pc,
    )

    if args.report:
        print("saving report")
        report = wb.Report(
            project=args.wandb_project_name,
            title=f"Regression Report: {exp_name}",
            description=str(args.filters),
            blocks=blocks,
        )
        report.save()
        print(f"view the generated report at {report.url}")
