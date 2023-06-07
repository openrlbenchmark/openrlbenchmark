import copy
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, cast
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
class PlotConfig:
    ncols: int = 2
    """the number of columns in the chart"""
    nrows: tyro.conf.Suppress[int] = None
    """(TO BE FILLED in runtime) the number of rows in the chart"""
    ncols_legend: int = 2
    """the number of legend columns in the chart"""
    xlabel: str = "Step"
    """the label of the x-axis"""
    ylabel: str = "Episodic Return"
    """the label of the y-axis"""
    sharex: bool = False
    """if toggled, we will share the x-axis across all subplots"""
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
    nsubsamples: int = 20
    """the number of subsamples to take from the wandb runs for IQM"""


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
    wandb_project_name: str = "cleanrl"
    """the wandb project name for the report creation"""
    offline: bool = False
    """if toggled, we will use the offline database instead of wandb"""
    pc: PlotConfig = field(default_factory=PlotConfig)
    """the plot configuration"""
    rliable: bool = False
    """if toggled, we will use rliable to compute the metrics"""


class Runset:
    def __init__(
        self,
        name: str,
        entity: str,
        project: str,
        metric: str = "charts/episodic_return",
        groupby: str = "",
        custom_exp_name_key: str = "exp_name",
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
        self.wandb_filters = {
            "$and": [
                {f"config.{self.custom_env_id_key}.value": self.env_id},
                *include_tag_groups,
                *user,
                {f"config.{self.custom_exp_name_key}.value": self.exp_name},
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


def to_rich_table(df: pd.DataFrame) -> Table:
    table = Table()
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    return table


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
    hns_result_table = pd.DataFrame(index=env_ids, columns=[runsets[0].name for runsets in runsetss])
    min_num_seeds_per_hypothesis = {}
    for runsets in runsetss:
        min_num_seeds_per_hypothesis[runsets[0].name] = float("inf")
    exs = []
    runtimes = []
    for idx, env_id in enumerate(env_ids):
        print(f"collecting runs for {env_id}")
        ex = expt.Experiment("Comparison")
        for runsets in runsetss:
            hypo = create_hypothesis(runsets[idx], scan_history)
            ex.add_hypothesis(hypo)
        exs.append(ex)

        # for each run `i` get the average of the last `rolling` episodes as r_i
        # then take the average and std of r_i as the results.
        result = []
        hns_result = []
        for hypothesis in ex.hypotheses:
            metric_result = []
            hns_metric_result = []
            console.print(f"{hypothesis.name} has {len(hypothesis.runs)} runs", style="bold")
            min_num_seeds_per_hypothesis[hypothesis.name] = min(
                min_num_seeds_per_hypothesis[hypothesis.name], len(hypothesis.runs)
            )
            for run in hypothesis.runs:
                # HACK: handle different Atari env id types.
                standard_env_id = env_id
                if env_id.endswith("NoFrameskip-v4"):
                    standard_env_id = env_id.replace("NoFrameskip-v4", "-v5")
                run.df["hns"] = (run.df["charts/episodic_return"] - atari_hns[standard_env_id][0]) / (
                    atari_hns[standard_env_id][1] - atari_hns[standard_env_id][0]
                )
                metric_result += [run.df["charts/episodic_return"].dropna()[-metric_last_n_average_window:].mean()]
                hns_metric_result += [run.df["hns"].dropna()[-metric_last_n_average_window:].mean()]

                # convert time unit in place
                if pc.time_unit == "m":
                    run.df["_runtime"] /= 60
                elif pc.time_unit == "h":
                    run.df["_runtime"] /= 3600
            metric_result = np.array(metric_result)
            hns_metric_result = np.array(hns_metric_result)
            result += [f"{metric_result.mean():.2f} ± {metric_result.std():.2f}"]
            hns_result += [f"{hns_metric_result.mean():.2f} ± {hns_metric_result.std():.2f}"]
        result_table.loc[env_id] = result
        hns_result_table.loc[env_id] = hns_result
        runtimes.append(list(ex.summary()["_runtime"]))
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
        ax2 = ax.twinx()
        ax2.set_ylim([0, ex.summary()["hns"].max()])
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
        ax_time2 = ax_time.twinx()
        ax_time2.set_ylim([0, ex.summary()["hns"].max()])

    runtimes = pd.DataFrame(np.array(runtimes), index=env_ids, columns=list(ex.summary()["name"]))
    console.rule(f"[bold red]Runtime ({pc.time_unit}) (mean ± std)")
    console.print(to_rich_table(runtimes.rename_axis("Environment").reset_index()))

    # for each run set, for each seed, plot 57 curves and get their median curves, then plot the average of the median curves
    hns_ex = expt.Experiment("Human Normalized Score")
    hns_dict = {}
    max_global_steps = defaultdict(int)
    for runsets_idx, runsets in enumerate(runsetss):
        hns_dict[runsets[0].name] = np.zeros((min_num_seeds_per_hypothesis[runsets[0].name], len(env_ids), pc.nsubsamples))
        # for each seed
        median_curves = []
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
                    min(run_of_one_seed.df["global_step"]), max(run_of_one_seed.df["global_step"]), num=pc.nsubsamples
                )
                hns_dict[runsets[0].name][seed_idx, ex_idx, :] = np.interp(
                    x_samples, run_of_one_seed.df["global_step"], run_of_one_seed.df["hns"]
                )

            temp_fig, temp_axe = plt.subplots(nrows=1, ncols=1)
            temp_hypo = Hypothesis("seed", runs_of_one_seed)
            temp_hypo.plot(
                ax=temp_axe,
                title="Human Normalized Score",
                x="global_step",
                y="hns",
                err_style="band",
                rolling=pc.rolling,
                n_samples=10000,
                representative_fn=lambda h: cast(pd.DataFrame, h.grouped.median()),  # median curve
                legend=False,
            )
            assert len(temp_axe.lines) == 1
            median_of_runs_of_one_seed = temp_axe.lines[0].get_xydata()
            print(f"min_global_step: {min_global_step}")
            max_global_steps[runsets[0].name] = max(max_global_steps[runsets[0].name], min_global_step)
            median_curves.append(median_of_runs_of_one_seed[median_of_runs_of_one_seed[:, 0] < min_global_step])
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
        err_style="unit_traces",
        std_alpha=0.1,
        n_samples=10000,
        rolling=pc.rolling,
        colors=[runsets[0].color for runsets in runsetss],
        legend=False,
    )
    hns_ex_time = copy.deepcopy(hns_ex)
    for hypo_idx, hypo in enumerate(hns_ex_time.hypotheses):
        for run_idx in range(len(hypo.runs)):
            hypo.runs[run_idx].df["_runtime"] = np.linspace(
                0, runtimes.mean(axis=0).iloc[hypo_idx], len(hypo.runs[run_idx].df)
            )

    axes_median_hns[0].set_ylabel("Median Human Normalized Score")
    axes_median_hns[0].set_xlabel(f"Steps")
    hns_ex_time.plot(
        title=" ",
        ax=axes_median_hns[1],
        x="_runtime",
        y="charts/episodic_return",
        err_style="unit_traces",
        std_alpha=0.1,
        n_samples=10000,
        rolling=pc.rolling,
        colors=[runsets[0].color for runsets in runsetss],
        legend=False,
    )
    axes_median_hns[1].set_xlabel(f"Time ({pc.time_unit})")
    h, l = axes_median_hns[0].get_legend_handles_labels()
    fig_median_hns.legend(
        h, l, loc="lower center", ncol=pc.ncols_legend, bbox_to_anchor=(0.5, 0.9), bbox_transform=fig_median_hns.transFigure
    )
    fig_median_hns.savefig(f"{output_filename}_hns_median.png", bbox_inches="tight")
    fig_median_hns.savefig(f"{output_filename}_hns_median.pdf", bbox_inches="tight")
    with open(f"{output_filename}_hns_median.pkl", "wb") as f:
        pickle.dump(
            [(line.get_color(), line.get_alpha(), line.get_xydata(), line.get_label()) for line in axes_median_hns[0].lines], f
        )
    console.rule(f"[bold red]{pc.ylabel} (mean ± std)")
    console.print(to_rich_table(result_table.rename_axis("Environment").reset_index()))
    result_table.to_markdown(open(f"{output_filename}.md", "w"))
    result_table.to_csv(open(f"{output_filename}.csv", "w"))

    console.rule(f"[bold red]Human-noramlized Score (mean ± std)")
    console.print(to_rich_table(hns_result_table.rename_axis("Environment").reset_index()))
    hns_result_table.to_markdown(open(f"{output_filename}_hns.md", "w"))
    hns_result_table.to_csv(open(f"{output_filename}_hns.csv", "w"))
    runtimes.to_markdown(open(f"{output_filename}_runtimes.md", "w"))
    runtimes.to_csv(open(f"{output_filename}_runtimes.csv", "w"))
    console.rule(f"[bold red]Runtime ({pc.time_unit}) Average")
    average_runtime = pd.DataFrame(runtimes.mean(axis=0)).reset_index()
    average_runtime.columns = ["Environment", "Average Runtime"]
    console.print(to_rich_table(average_runtime))

    # add legend
    h, l = axes_flatten[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=pc.ncols_legend, bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure)
    fig.supxlabel(pc.xlabel)
    fig.supylabel(pc.ylabel)
    fig.text(0.99, 0.5, "Human-Normalized Score", va="center", rotation=-90)
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
    if os.path.dirname(output_filename) != "":
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fig.subplots_adjust(hspace=pc.hspace, wspace=pc.wspace)
    fig_time.subplots_adjust(hspace=pc.hspace, wspace=pc.wspace)
    fig.savefig(f"{output_filename}.png", bbox_inches="tight")
    fig.savefig(f"{output_filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_filename}.svg", bbox_inches="tight")
    fig_time.savefig(f"{output_filename}-time.png", bbox_inches="tight")
    fig_time.savefig(f"{output_filename}-time.pdf", bbox_inches="tight")
    fig_time.savefig(f"{output_filename}-time.svg", bbox_inches="tight")
    return blocks, hns_dict, max_global_steps


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
        pprint(
            {
                "wandb_project_name": wandb_project_name,
                "wandb_entity": wandb_entity,
                "custom_env_id_key": custom_env_id_key,
                "custom_exp_name_key": custom_exp_name_key,
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
                    assert len(runsets[0].runs) > 0, f"{exp_name} ({query}) in {env_id} has no runs"
            runsetss.append(runsets)

    blocks, hns_dict, max_global_steps = compare(
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
        print("plotting sample efficiency curve")
        exp_names = list(reversed(list(hns_dict.keys())))
        colors_flatten = colors_flatten_original
        colors = dict(zip(list(hns_dict.keys()), colors_flatten))
        frames = np.linspace(0, max(max_global_steps.values()), args.pc.nsubsamples)
        fig_rly_hns, axes_rly_hns = plt.subplots(ncols=2, figsize=(7 * 2, 3.4))
        iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])])
        iqm_scores, iqm_cis = rly.get_interval_estimates(hns_dict, iqm, reps=50000)
        plot_utils.plot_sample_efficiency_curve(
            frames + 1,
            iqm_scores,
            iqm_cis,
            algorithms=exp_names,
            colors=colors,
            xlabel=r"Steps",
            ax=axes_rly_hns[0],
            ylabel="IQM Human Normalized Score",
            labelsize="x-large",
            ticklabelsize="x-large",
        )
        expt.plot.autoformat_xaxis(axes_rly_hns[0])

        print("plotting performance profiles")
        atari_200m_thresholds = np.linspace(0.0, 8.0, 81)
        atari_200m_normalized_score_dict = {}
        for key, value in hns_dict.items():
            atari_200m_normalized_score_dict[key] = np.nanmean(value[:, :, -1:], axis=-1)
        score_distributions, score_distributions_cis = rly.create_performance_profile(
            atari_200m_normalized_score_dict, atari_200m_thresholds
        )
        plot_utils.plot_performance_profiles(
            score_distributions,
            atari_200m_thresholds,
            performance_profile_cis=score_distributions_cis,
            colors=colors,
            xlabel=r"Human Normalized Score $(\tau)$",
            ax=axes_rly_hns[1],
        )
        fig_rly_hns.savefig(f"{args.output_filename}_iqm_profile.png", bbox_inches="tight")
        fig_rly_hns.savefig(f"{args.output_filename}_iqm_profile.pdf", bbox_inches="tight")

        print("plotting aggregate metrics")
        aggregate_func = lambda x: np.array(
            [
                metrics.aggregate_median(x),
                metrics.aggregate_iqm(x),
                metrics.aggregate_mean(x),
                metrics.aggregate_optimality_gap(x),
            ]
        )
        aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
            atari_200m_normalized_score_dict, aggregate_func, reps=50000
        )
        # make aggregate_scores into a dataframe
        aggregate_scores_df = pd.DataFrame.from_dict(
            aggregate_scores, orient="index", columns=["Median", "IQM", "Mean", "Optimality Gap"]
        )
        print("aggregate_scores", aggregate_scores_df)
        fig, axes = plot_utils.plot_interval_estimates(
            aggregate_scores,
            aggregate_score_cis,
            metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
            algorithms=exp_names,
            colors=colors,
            # xlabel='Human Normalized Score',
            xlabel="",
        )
        plt.savefig(f"{args.output_filename}_hns_aggregate.png", bbox_inches="tight")
        plt.savefig(f"{args.output_filename}_hns_aggregate.pdf", bbox_inches="tight")

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
