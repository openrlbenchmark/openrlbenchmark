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
        metric: str = "charts/episodic_return",
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
        self.metric = metric
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


def create_hypothesis(runset: Runset, scan_history: bool = False) -> Hypothesis:
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
        if runset.metric not in run_df:
            print(f"Skipping run {run} because metric {runset.metric} not found")
            continue
        if len(runset.metric) > 0:
            run_df["charts/episodic_return"] = run_df[runset.metric]
        cleaned_df = run_df[["global_step", "_runtime", "charts/episodic_return"]].dropna()
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
            metric_over_step = wb.LinePlot(
                x=runsets[idx].custom_xaxis_key,
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
                    {
                        (
                            runsets[idx].report_runset.name,
                            runsets[idx].runs[0].config[runsets[idx].custom_exp_name_key],
                        ): runsets[idx].color
                    }
                )
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
        hypotheses = [create_hypothesis(runsets[idx], scan_history) for runsets in runsetss]
        ex = expt.Experiment("Comparison", hypotheses)
        exs.append(ex)

        # for each run `i` get the average of the last `rolling` episodes as r_i
        # then take the average and std of r_i as the results.
        result = []
        for hypothesis in ex.hypotheses:
            metric_result = []
            console.print(f"{hypothesis.name} has {len(hypothesis.runs)} runs", style="bold")
            for run in hypothesis.runs:
                metric_result += [run.df["charts/episodic_return"].dropna()[-metric_last_n_average_window:].mean()]

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
        ax = axes_flatten[idx]
        ex.plot(
            ax=ax,
            title=env_id,
            x="global_step",
            y="charts/episodic_return",
            err_style="band",
            std_alpha=0.1,
            n_samples=10000,
            rolling=pc.rolling,
            colors=[runsets[idx].color for runsets in runsetss],
            legend=False,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        if pc.max_steps is not None:
            ax.set_xlim(0, pc.max_steps)
        ax_time = axes_time_flatten[idx]
        ex.plot(
            ax=ax_time,
            title=env_id,
            x="_runtime",
            y="charts/episodic_return",
            err_style="band",
            std_alpha=0.1,
            n_samples=10000,
            rolling=pc.rolling,
            colors=[runsets[idx].color for runsets in runsetss],
            legend=False,
        )
        ax_time.set_xlabel("")
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
    fig.supylabel(pc.ylabel)
    fig.tight_layout()
    h, l = axes_time_flatten[0].get_legend_handles_labels()
    fig_time.legend(
        h, l, loc="lower center", ncol=pc.ncols_legend, bbox_to_anchor=(0.5, 1.0), bbox_transform=fig_time.transFigure
    )
    fig_time.supxlabel(f"Time ({pc.time_unit})")
    fig_time.supylabel(pc.ylabel)
    fig_time.tight_layout()

    # remove the empty axes
    for ax in axes_flatten[len(env_ids) :]:
        ax.remove()
    for ax in axes_time_flatten[len(env_ids) :]:
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
    # calculate the number of rows
    args.pc.nrows = np.ceil(len(args.env_ids[0]) / args.pc.ncols).astype(int)

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
        metric = query["metric"][0] if "metric" in query else "charts/episodic_return"
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
                "metric": metric,
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
                        metric=metric,
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

    if args.rliable:
        # get min num seeds per hypothesis
        min_num_seeds_per_hypothesis = {}
        for runsets in runsetss:
            min_num_seeds_per_hypothesis[runsets[0].name] = float("inf")
        for ex in exs:
            for hypothesis in ex.hypotheses:
                console.print(f"{hypothesis.name} has {len(hypothesis.runs)} runs", style="bold")
                min_num_seeds_per_hypothesis[hypothesis.name] = min(
                    min_num_seeds_per_hypothesis[hypothesis.name], len(hypothesis.runs)
                )

        # create `score_dict`; each item in `score_dict` has shape (num_seeds, len(args.env_ids[0]), nsubsamples)
        score_dict = {}
        max_global_steps = defaultdict(int)
        for runsets_idx, runsets in enumerate(runsetss):
            score_dict[runsets[0].name] = np.zeros(
                (min_num_seeds_per_hypothesis[runsets[0].name], len(args.env_ids[0]), args.rc.nsubsamples)
            )
            # for each seed
            for seed_idx, _ in enumerate(range(min_num_seeds_per_hypothesis[runsets[0].name])):  # exs[0][runsets_idx]
                min_global_step = float("inf")
                print(f"collecting runs for {runsets[0].name} seed {seed_idx}")

                runs_of_one_seed = []
                for ex_idx, ex in enumerate(exs):
                    run_of_one_seed = ex[runsets_idx][seed_idx]
                    min_global_step = min(min_global_step, run_of_one_seed.df["global_step"].iloc[-1])
                    runs_of_one_seed.append(run_of_one_seed)

                    # interpolate
                    x_samples = np.linspace(
                        min(run_of_one_seed.df["global_step"]), max(run_of_one_seed.df["global_step"]), num=args.rc.nsubsamples
                    )
                    score_dict[runsets[0].name][seed_idx, ex_idx, :] = np.interp(
                        x_samples, run_of_one_seed.df["global_step"], run_of_one_seed.df["charts/episodic_return"]
                    )
                max_global_steps[runsets[0].name] = max(max_global_steps[runsets[0].name], min_global_step)

        exp_names = list(reversed(list(score_dict.keys())))
        colors_flatten = colors_flatten_original
        colors = dict(zip(list(score_dict.keys()), colors_flatten))
        frames = np.linspace(0, max(max_global_steps.values()), args.rc.nsubsamples)
        print_rich_table(
            f"Items in the `score_dict` used for `rliable`",
            pd.DataFrame(
                data=[score_dict[key].shape for key in score_dict],
                columns=["Number of Seeds", "Number of Environments", "Number of Sub-samples"],
                index=list(score_dict.keys()),
            )
            .rename_axis("Experiments")
            .reset_index(),
            console,
        )

        # normalize scores.
        if args.rc.score_normalization_method == "maxmin":
            normalized_score_dict = maxmin_normalize_score(score_dict)
        elif args.rc.score_normalization_method == "atari":
            normalized_score_dict = atari_normalize_score(score_dict, args.env_ids[0])
        else:
            raise NotImplementedError(f"Normalization method {args.rc.score_normalization_method} not implemented")
        performance_profile_normalized_score_dict = {}
        for key, value in normalized_score_dict.items():
            performance_profile_normalized_score_dict[key] = np.nanmean(value[:, :, -1:], axis=-1)
        metric_fns = [
            metrics.aggregate_median,
            metrics.aggregate_iqm,
            metrics.aggregate_mean,
            metrics.aggregate_optimality_gap,
        ]
        metric_names = ["Median", "IQM", "Mean", "Optimality Gap"]

        if args.rc.sample_efficiency_plots:
            print("plotting sample efficiency curve (this is slow and may take several minutes)")
            fig_sample_efficiency, axes_sample_efficiency = plt.subplots(
                ncols=2,
                nrows=2,
                figsize=(7 * 2, 3.4 * 2),
                sharex=args.pc.sharex,
            )
            for metric_fn, ax, metric_name in zip(metric_fns, axes_sample_efficiency.flatten(), metric_names):
                aggregate_fn = lambda scores: np.array([metric_fn(scores[..., frame]) for frame in range(scores.shape[-1])])
                aggregate_scores, aggregate_cis = rly.get_interval_estimates(
                    normalized_score_dict, aggregate_fn, reps=args.rc.sample_efficiency_num_bootstrap_reps
                )
                for exp_name in score_dict.keys():
                    global_step = global_steps[exp_name].mean()
                    global_step_xaxis = np.linspace(0, global_step, args.rc.nsubsamples)
                    plot_utils.plot_sample_efficiency_curve(
                        global_step_xaxis,
                        {exp_name: aggregate_scores[exp_name]},
                        {exp_name: aggregate_cis[exp_name]},
                        algorithms=[exp_name],
                        colors=colors,
                        xlabel=r"Steps",
                        ax=ax,
                        ylabel=metric_name,
                        labelsize="x-large",
                        ticklabelsize="x-large",
                    )
                ax.set_xlabel("")
                expt.plot.autoformat_xaxis(ax)

                if metric_name == args.rc.sample_efficiency_and_walltime_efficiency_method:
                    fig_median_sample_walltime_efficiency, axes_median_sample_walltime_efficiency = plt.subplots(
                        ncols=2,
                        figsize=(7 * 2, 3.4),
                        sharey=True,
                    )
                    for exp_name in score_dict.keys():
                        global_step = global_steps[exp_name].mean()
                        global_step_xaxis = np.linspace(0, global_step, args.rc.nsubsamples)
                        plot_utils.plot_sample_efficiency_curve(
                            global_step_xaxis,
                            {exp_name: aggregate_scores[exp_name]},
                            {exp_name: aggregate_cis[exp_name]},
                            algorithms=[exp_name],
                            colors=colors,
                            xlabel=r"Steps",
                            ax=axes_median_sample_walltime_efficiency[0],
                            ylabel=metric_name,
                            labelsize="x-large",
                            ticklabelsize="x-large",
                        )
                    expt.plot.autoformat_xaxis(axes_median_sample_walltime_efficiency[0])
                    for exp_name in score_dict.keys():
                        runtime = runtimes[exp_name].mean()
                        runtime_xaxis = np.linspace(0, runtime, args.rc.nsubsamples)
                        plot_utils.plot_sample_efficiency_curve(
                            runtime_xaxis,
                            {exp_name: aggregate_scores[exp_name]},
                            {exp_name: aggregate_cis[exp_name]},
                            algorithms=[exp_name],
                            colors=colors,
                            xlabel=f"Time ({args.pc.time_unit})",
                            ax=axes_median_sample_walltime_efficiency[1],
                            ylabel=metric_name,
                            labelsize="x-large",
                            ticklabelsize="x-large",
                        )
                    axes_median_sample_walltime_efficiency[1].set_ylabel("")
                    h, l = axes_median_sample_walltime_efficiency[1].get_legend_handles_labels()
                    fig_median_sample_walltime_efficiency.legend(
                        h,
                        l,
                        loc="lower center",
                        ncol=args.pc.ncols_legend,
                        bbox_to_anchor=(0.5, 1.0),
                        bbox_transform=fig_median_sample_walltime_efficiency.transFigure,
                    )
                    fig_median_sample_walltime_efficiency.tight_layout()
                    fig_median_sample_walltime_efficiency.savefig(
                        f"{args.output_filename}_sample_walltime_efficiency.png", bbox_inches="tight"
                    )
                    fig_median_sample_walltime_efficiency.savefig(
                        f"{args.output_filename}_sample_walltime_efficiency.pdf", bbox_inches="tight"
                    )

            h, l = axes_sample_efficiency[0][0].get_legend_handles_labels()
            fig_sample_efficiency.legend(
                h,
                l,
                loc="lower center",
                ncol=args.pc.ncols_legend,
                bbox_to_anchor=(0.5, 1.0),
                bbox_transform=fig_sample_efficiency.transFigure,
            )
            fig_sample_efficiency.supxlabel(args.pc.xlabel, fontsize="x-large")
            fig_sample_efficiency.tight_layout()
            fig_sample_efficiency.savefig(f"{args.output_filename}_sample_efficiency.png", bbox_inches="tight")
            fig_sample_efficiency.savefig(f"{args.output_filename}_sample_efficiency.pdf", bbox_inches="tight")

        if args.rc.performance_profile_plots:
            print("plotting performance profiles")
            fig_performance_profile, axes_performance_profile = plt.subplots(
                ncols=2,
                figsize=(7 * 2, 3.4),
            )
            performance_profile_thresholds = np.linspace(0.0, args.rc.normalized_score_threshold, 81)
            score_distributions, score_distributions_cis = rly.create_performance_profile(
                performance_profile_normalized_score_dict,
                performance_profile_thresholds,
                reps=args.rc.performance_profile_num_bootstrap_reps,
            )
            avg_score_distributions, avg_score_distributions_cis = rly.create_performance_profile(
                performance_profile_normalized_score_dict,
                performance_profile_thresholds,
                reps=args.rc.performance_profile_num_bootstrap_reps,
                use_score_distribution=False,
            )
            plot_utils.plot_performance_profiles(
                score_distributions,
                performance_profile_thresholds,
                performance_profile_cis=score_distributions_cis,
                colors=colors,
                xlabel=r"Normalized Score $(\tau)$",
                ax=axes_performance_profile[0],
            )
            plot_utils.plot_performance_profiles(
                avg_score_distributions,
                performance_profile_thresholds,
                performance_profile_cis=avg_score_distributions_cis,
                colors=colors,
                xlabel=r"Normalized Score $(\tau)$",
                ylabel=r"Fraction of tasks with score > $\tau$",
                ax=axes_performance_profile[1],
            )
            h, l = axes_performance_profile[0].get_legend_handles_labels()
            fig_performance_profile.legend(
                h,
                l,
                loc="lower center",
                ncol=args.pc.ncols_legend,
                bbox_to_anchor=(0.5, 1.0),
                bbox_transform=fig_performance_profile.transFigure,
            )
            fig_performance_profile.tight_layout()
            fig_performance_profile.savefig(f"{args.output_filename}_performance_profile.png", bbox_inches="tight")
            fig_performance_profile.savefig(f"{args.output_filename}_performance_profile.pdf", bbox_inches="tight")

        if args.rc.aggregate_metrics_plots:
            print("plotting aggregate metrics")
            aggregate_func = lambda x: np.array([metric_fn(x) for metric_fn in metric_fns])
            aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
                performance_profile_normalized_score_dict, aggregate_func, reps=args.rc.interval_estimates_num_bootstrap_reps
            )
            aggregate_scores_df = pd.DataFrame.from_dict(
                aggregate_scores, orient="index", columns=["Median", "IQM", "Mean", "Optimality Gap"]
            )
            print_rich_table(f"Aggregate Scores", aggregate_scores_df.reset_index(), console)
            fig, axes = plot_utils.plot_interval_estimates(
                aggregate_scores,
                aggregate_score_cis,
                metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
                algorithms=exp_names,
                colors=colors,
                xlabel="",
                # xlabel='Normalized Score',
                # xlabel_y_coordinate=-0.08, # this variable needs to be adjusted for each plot... :( so we just disable xlabel for now.
            )
            axes[1].set_xlabel("Normalized Score", fontsize="xx-large")
            fig.tight_layout()
            plt.savefig(f"{args.output_filename}_aggregate.png", bbox_inches="tight")
            plt.savefig(f"{args.output_filename}_aggregate.pdf", bbox_inches="tight")

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
