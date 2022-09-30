import matplotlib.pyplot as plt
import wandb
from expt import Hypothesis, Run

api = wandb.Api()
wandb_runs = api.runs(
    path="openrlbenchmark/cleanrl",
    filters={"$and": [{"config.env_id.value": "Walker2d-v2"}, {"config.exp_name.value": "ddpg_continuous_action"}]},
)

runs = []
for idx, run in enumerate(wandb_runs):
    wandb_run = run.history(keys=["charts/episodic_return", "charts/episodic_length", "global_step"])
    del wandb_run["_step"]
    wandb_run = wandb_run.set_index("global_step")
    runs += [Run(f"seed{idx}", wandb_run)]

hypothesis = Hypothesis(name="hypo_plot", runs=runs)
g = hypothesis.plot(y="charts/episodic_return", n_samples=400, rolling=50)
plt.savefig("test.png")
