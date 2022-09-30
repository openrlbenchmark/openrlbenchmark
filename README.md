# Open RL Benchmark: Comprehensive Tracked Experiments for Reinforcement Learning

Open RL Benchmark is a comprehensive collection of tracked experiments for RL. It aims to make it easier for RL practitioners to pull and compare all kinds of metrics from reputable RL libraries like Stable-baselines3, Tianshou, CleanRL, and others.

Check out this [google doc](https://docs.google.com/document/d/1cDI_AMr2QVmkC53dCHFMYwGJtLC8V4p6KdL2wnYPaiI/edit?usp=sharing) for more info and comment.


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
poetry run pythonbaselines_atari_hns.py
```

which will generate images in the `static` folder, such as 

![](static/hms_each_game.svg)


and the following reports:

* [Atari: openai/baselins' PPO (part 1)](https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-openai-baselins-PPO-part-1---VmlldzoyNzIyNzg2)
* [Atari: openai/baselins' PPO (part 2)](https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-openai-baselins-PPO-part-2---VmlldzoyNzIyNzg3)


## Get started

Check out the [Open RL Benchmark reports](https://wandb.ai/openrlbenchmark/openrlbenchmark/reportlist): featuring W&B reports with tracked Atari, MuJoCo experiments from SB3, CleanRL, and others.

![DEMO](https://user-images.githubusercontent.com/5555347/167724483-3c038a3b-3dce-4aa9-8cf0-6cedae52d321.gif)

You can "fork" these reports and use them in your own workspace. See the following video for a demo, where I used a _newly-created_ W&B account to clone a report from [Open RL Benchmark reports](https://wandb.ai/openrlbenchmark/openrlbenchmark/reportlist) and compared my metrics with the report's metrics.


https://user-images.githubusercontent.com/5555347/167722421-7f6a138e-6374-491a-8d6e-3b0604e73884.mp4

## What's going on right now?

This is a project we are slowly working on. We have already added all benchmark experiments from CleanRL and @araffin is working on adding SB3 benchmark experiments. We don't have a defined timeline yet, but if you want to get involved. Feel free to reach out to me or open an issue.
