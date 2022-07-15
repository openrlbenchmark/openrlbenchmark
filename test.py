results_url = "https://openaisciszymon.blob.core.windows.net/dqn-blogpost/results.pkl"
download_path = "dqn_results.pkl"
with open(download_path, "rb") as f:
    import pickle
    run_to_episode_data = pickle.load(f)

import multiprocess
import os
import pandas as pd

def process_data():
    for k in sorted(run_to_episode_data.keys()):
        df = pd.DataFrame(run_to_episode_data[k]['episode_data'])
        os.makedirs(f"data/{k}", exist_ok=True)
        df.to_csv(f"data/{k}/progress.csv", index=False)

    print(f"Total number of directories = {len(run_to_episode_data)}")

if not os.path.isdir("data/"):
    process_data()

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image

mpl.rcParams['pdf.fonttype'] = 42     # use true-type
mpl.rcParams['ps.fonttype'] = 42      # use true-type
mpl.rcParams['font.size'] = 12

import expt
print("expt:", expt.__version__)

import pandas as pd
import numpy as np

run = expt.get_runs("./data/atari-prior-duel-a-alien-*")
print(run[0])
print(run[0].df
)
# run[0].plot(rolling=100)
# plt.savefig("test.png")

ex = expt.Experiment("Comparison of DQN Variants")
env = 'amidar'

def make_total_timesteps(r: expt.Run) -> expt.Run:
    """A post-processing after parsing the run: Add a new column 'total_timesteps'.
    
    Note that in this example, the raw CSV logging data did not include global step (total_timesteps).
    So we take a cumulative sum of episode lengths, which can be the x-axis (total environment steps) for all plots below.
    """
    
    # This should be our x-axis. Note that this might be not aligned across different runs
    r.df['total_timesteps'] = r.df['episode_lengths'].cumsum()
    return r

for algo, desc in {"a": "DQN",
                   "duel-a": "Double DQN",
                   "prior-a": "DQN + PER",
                   "prior-duel-a": "Double DQN + PER",
                  }.items():
    runs = expt.get_runs(f"./data/atari-{algo}-{env}-*", progress_bar=None,
                         run_postprocess_fn=make_total_timesteps,
                         pool_class=multiprocess.Pool)
    
    # h = runs.to_hypothesis(name=desc)
    # ex.add_hypothesis(desc, h)
    ex.add_runs(desc, runs=runs)

# Summarize the data: by default, the average over last 10% rows are reported.
print(ex.summary())

g = ex.plot(x='total_timesteps', y=['episode_rewards', 'episode_lengths'],
            rolling=50, n_samples=1000, legend='episode_lengths', figsize=(10, 4))

plt.savefig("test.png")