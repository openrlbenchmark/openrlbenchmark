import enum
from matplotlib import axis
import numpy as np
import expt
from expt import Run, Hypothesis, Experiment
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import wandb.apis.reports as wb  # noqa


api = wandb.Api()
wandb_runs = api.runs(
    path="openrlbenchmark/cleanrl",
    filters={'$and': [{'config.env_id.value': 'Walker2d-v2'}, {'config.exp_name.value': 'ddpg_continuous_action'}]}
)
wandb_runs2 = api.runs(
    path="openrlbenchmark/cleanrl",
    filters={'$and': [{'config.env_id.value': 'Walker2d-v2'}, {'config.exp_name.value': 'ddpg_continuous_action_jax'}]}
)

def create_hypothesis(name, wandb_runs):
    
    runs = []
    for idx, run in enumerate(wandb_runs):
        wandb_run = run.history()
        if 'videos' in wandb_run:
            wandb_run = wandb_run.drop(columns=['videos'], axis=1)
        runs += [Run(f"seed{idx}", wandb_run)]
    return Hypothesis(name, runs)


# g = hypothesis.plot(y="charts/episodic_return", x="global_step", n_samples=400, rolling=50)


ex = expt.Experiment("Comparison of DQN Variants")
h = create_hypothesis("DDPG + torch", wandb_runs)
ex.add_hypothesis(h)
ex.add_hypothesis(create_hypothesis("DDPG + jax", wandb_runs2))
ex.plot(x="global_step", n_samples=400, rolling=50, err_style="fill")
plt.savefig("test.png")

# r = wandb_runs[0].history()