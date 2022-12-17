# Open RL Benchmark: Comprehensive Tracked Experiments for Reinforcement Learning

Open RL Benchmark is a comprehensive collection of tracked experiments for RL. It aims to make it easier for RL practitioners to pull and compare all kinds of metrics from reputable RL libraries like Stable-baselines3, Tianshou, CleanRL, and others.

* 📜 [Design docs](https://docs.google.com/document/d/1cDI_AMr2QVmkC53dCHFMYwGJtLC8V4p6KdL2wnYPaiI/edit?usp=sharing): check out our motivation and vision.
* 🔗 [Open RL Benchmark reports](https://wandb.ai/openrlbenchmark/openrlbenchmark/reportlist): featuring W&B reports with tracked Atari, MuJoCo experiments from SB3, CleanRL, and others.

> ⚠️ There has been a major refactoring. If you are looking for older version, check out [43fc8e2](https://github.com/openrlbenchmark/openrlbenchmark/tree/43fc8e2066ac6371913ac53b629928ac15a65e13)

## Get started

Prerequisites:
* Python >=3.7.1,<3.10 (not yet 3.10)
* [Poetry 1.2.1+](https://python-poetry.org)

Open RL Benchmark provides an RLops API to pull and compare metrics from Weights and Biases. The following example shows how to compare the performance of SB3's ppo, a2c, ddpg, ppo_lstm, sac, td3, ppo, trpo, CleanRL's sac on HalfCheetahBulletEnv-v0.

```
poetry run python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=sb3&ceik=env&cen=algo&metric=rollout/ep_rew_mean' \
        'a2c' \
        'ddpg' \
        'ppo_lstm' \
        'sac' \
        'td3' \
        'ppo' \
        'trpo' \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'sac_continuous_action?tag=rlops-pilot' \
    --env-ids HalfCheetahBulletEnv-v0 \
    --ncols 1 \
    --ncols-legend 2 \
    --output-filename compare.png \
    --report
```

Here, we use create multiple filters. The first string in the first filter is `'?we=openrlbenchmark&wpn=sb3&ceik=env&cen=algo&metric=rollout/ep_rew_mean'`, which is a query string that specifies the following:

* `we`: the W&B entity name
* `wpn`: the W&B project name
* `ceik`: the custom key for the environment id
* `cen`: the custom key for the experiment name

So we are fetching metrics from [https://wandb.ai/openrlbenchmark/sb3](https://wandb.ai/openrlbenchmark/sb3). The environment id is stored in the `env` key, and the experiment name is stored in the `algo` key. The metric we are interested in is `rollout/ep_rew_mean`.

Similary, we are fetching metrics from [https://wandb.ai/openrlbenchmark/cleanrl](https://wandb.ai/openrlbenchmark/cleanrl). The environment id is stored in the `env_id` key, and the experiment name is stored in the `exp_name` key. The metric we are interested in is `charts/episodic_return`.

The command above generates the following plot:

![](static/cleanrl_vs_sb3.png)

The `--report` tag also generates a [wandb report](https://wandb.ai/costa-huang/cleanrl/reports/Regression-Report-sac_continuous_action--VmlldzozMTY4NDQ3)

<!-- 
## Pre-alpha API


[`baselines_atari_hns.py`](https://github.com/openrlbenchmark/openrlbenchmark/blob/main/baselines_atari_hns.py) contains a pre-alpha API that demonstrates the matplotlib front end of the project, which has the following contents:

```python
import wandb.apis.reports as wb
from openrlbenchmark import Runset, plot_atari
blocks = plot_atari.plot_atari(
    [
        Runset(
            name="openai/baselines' PPO",
            filters=[{"config.exp_name.value": "baselines-ppo2-cnn"}],
            entity="openrlbenchmark",
            project="baselines",
            groupby="exp_name",
            key_for_env_id="config.env.value",
            x_axis="global_step",
            y_axis="charts/episodic_return",
            env_id_fn=lambda env_id: env_id.replace("-v5", "NoFrameskip-v4"),
        ),
    ],
    output_folder="static",
    return_wandb_report_blocks=True,
)
report = wb.Report(
    project="cleanrl",
    title="openai/baselins' PPO (part 1)",
    blocks=blocks[:29],
)
report.save()
print(f"view the generated report at {report.url}")
report = wb.Report(
    project="openrlbenchmark",
    title="openai/baselins' PPO (part 2)",
    blocks=blocks[29:],
)
report.save()
print(f"view the generated report at {report.url}")
```


> Note this API is not stable and may change in the future. Feel free to leave comments, suggestions, and make PRs.

To give it a run, please execute the following commands:

```bash
poetry install
poetry run python baselines_atari_hns.py
```

which will generate images in the `static` folder, such as 

![](static/hms_each_game.svg)


and the following reports:

* [Atari: openai/baselins' PPO (part 1)](https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-openai-baselins-PPO-part-1---VmlldzoyNzIyNzg2)
* [Atari: openai/baselins' PPO (part 2)](https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-openai-baselins-PPO-part-2---VmlldzoyNzIyNzg3)
 -->

## Get started

Check out the [Open RL Benchmark reports](https://wandb.ai/openrlbenchmark/openrlbenchmark/reportlist): featuring W&B reports with tracked Atari, MuJoCo experiments from SB3, CleanRL, and others.

![DEMO](https://user-images.githubusercontent.com/5555347/167724483-3c038a3b-3dce-4aa9-8cf0-6cedae52d321.gif)

You can "fork" these reports and use them in your own workspace. See the following video for a demo, where I used a _newly-created_ W&B account to clone a report from [Open RL Benchmark reports](https://wandb.ai/openrlbenchmark/openrlbenchmark/reportlist) and compared my metrics with the report's metrics.


https://user-images.githubusercontent.com/5555347/167722421-7f6a138e-6374-491a-8d6e-3b0604e73884.mp4

## What's going on right now?

This is a project we are slowly working on. We have already added all benchmark experiments from CleanRL and @araffin is working on adding SB3 benchmark experiments. We don't have a defined timeline yet, but if you want to get involved. Feel free to reach out to me or open an issue.
