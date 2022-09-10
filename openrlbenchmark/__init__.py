from typing import Callable, Dict, List
import wandb
from expt import Run
import wandb.apis.reports as wb  # noqa

class Runset:
    def __init__(
        self,
        name: str,
        filters: List[Dict],
        entity: str,
        project: str,
        groupby: str = "",
        key_for_env_id: str = "env_id",
        x_axis: str = "global_step",
        y_axis: str = "charts/avg_episodic_return",
        env_id_fn: Callable = lambda env_id: env_id,
    ):
        self.name = name
        self.filters = filters
        self.entity = entity
        self.project = project
        self.groupby = groupby
        self.key_for_env_id = key_for_env_id
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.env_id_fn = env_id_fn

    def runs(self, env_id: str):
        return wandb.Api().runs(
            path=f"{self.entity}/{self.project}",
            filters={"$and": [{self.key_for_env_id: self.env_id_fn(env_id)}] + self.filters}
        )

    def report_runset(self, env_id: str):
        return wb.RunSet(
            name=self.name,
            entity=self.entity,
            project=self.project,
            filters={"$or": [{"$and": [{self.key_for_env_id: self.env_id_fn(env_id)}] + self.filters}]},
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
