import expt
import matplotlib.pyplot as plt
import wandb
import wandb.apis.reports as wb  # noqa
from expt import Hypothesis, Run
from expt.plot import GridPlot

api = wandb.Api()


def create_hypothesis(name, wandb_runs):
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
            groupby=[self.groupby],
        )


if __name__ == "__main__":
    wandb.require("report-editing")

    env_ids = [
        # "Ant-v4",
        "HalfCheetah-v4",
        "Hopper-v4",
        "Humanoid-v4",
        "InvertedDoublePendulum-v4",
        "InvertedPendulum-v4",
        "Pusher-v4",
        "Reacher-v4",
        "Swimmer-v4",
        "Walker2d-v4",
    ]

    g = GridPlot(y_names=env_ids)
    blocks = []

    for env_id in env_ids:
        # NOTE: this is where to change things
        runset1 = Runset(
            name="CleanRL PPO + Envpool",
            filters={"$and": [{"config.env_id.value": env_id}, {"config.exp_name.value": "ppo_continuous_action_envpool"}]},
            entity="openrlbenchmark",
            project="envpool-cleanrl",
            groupby="exp_name",
        )
        runset2 = Runset(
            name="jaxrl's SAC",
            filters={
                "$and": [{"config.env_name.value": env_id.replace("v4", "v2")}, {"config.algo.value": "sac"}]
            },
            entity="openrlbenchmark",
            project="jaxrl",
            groupby="env_name",
        )

        l1 = wb.LinePlot(
            x="global_step",
            y=["charts/episodic_return", "training/return"],
            title=env_id,
            title_x="Steps",
            title_y="Episodic Return",
            max_runs_to_show=100,
            smoothing_factor=0.8,
            groupby_rangefunc="stderr",
            legend_template="${runsetName}",
        )
        l1.config["aggregateMetrics"] = True
        l2 = wb.LinePlot(
            x="_runtime",
            y=["charts/episodic_return", "training/return"],
            title=env_id,
            title_y="Episodic Return",
            max_runs_to_show=100,
            smoothing_factor=0.8,
            groupby_rangefunc="stderr",
            legend_template="${runsetName}",
        )
        l2.config["aggregateMetrics"] = True

        blocks += [
            wb.PanelGrid(
                runsets=[
                    runset1.report_runset,
                    runset2.report_runset,
                ],
                panels=[
                    l1,
                    l2,
                    # wb.MediaBrowser(
                    #     num_columns=2,
                    #     media_keys="videos",
                    # ),
                ],
            ),
        ]
        # ex = expt.Experiment("Comparison of PPO")
        # h = create_hypothesis(runset1.name, runset1.runs)
        # ex.add_hypothesis(h)
        # h2 = create_hypothesis(runset2.name, runset2.runs)
        # ex.add_hypothesis(h2)
        # ex.plot(
        #     ax=g[env_id],
        #     title=env_id,
        #     x="_runtime",
        #     y="charts/episodic_return",
        #     err_style="band",
        #     std_alpha=0.1,
        #     rolling=50,
        #     n_samples=400,
        #     legend=False,
        # )

    # g.add_legend(ax=g.axes[-1, -1], loc="upper left", bbox_to_anchor=(0, 1))
    # # Some post-processing with matplotlib API so that the plot looks nicer
    # for ax in g.axes_active:
    #     ax.xaxis.set_label_text("")
    #     ax.yaxis.set_label_text("")
    #     # ax.set_xlim(0, 200e6)
    # # g.fig.tight_layout(h_pad=1.3, w_pad=1.0)
    # print("saving figure to test.png")
    # plt.savefig("test.png")
    # print("saving report")
    report = wb.Report(
        project="cleanrl",
        title="MuJoCo CleanRL PPO vs OpenAI/Baselines PPO",
        blocks=blocks,
    )
    report.save()
    print(f"view the generated report at {report.url}")
