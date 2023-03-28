# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Costa: Note the data is not the same as Mnih et al., 2015
# Note the random agent score on Video Pinball is sometimes greater than the
# human score under other evaluation methods.


import argparse

import numpy as np
import pandas as pd
from rich import print


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=[],
        help="the csv files to calculate the median human normalized scores")
    # fmt: on
    return parser.parse_args()


atari_human_normalized_scores = {
    "Alien-v5": (227.8, 7127.7),
    "Amidar-v5": (5.8, 1719.5),
    "Assault-v5": (222.4, 742.0),
    "Asterix-v5": (210.0, 8503.3),
    "Asteroids-v5": (719.1, 47388.7),
    "Atlantis-v5": (12850.0, 29028.1),  # note our Envpool + PPO only gets 25 as the base return
    "BankHeist-v5": (14.2, 753.1),  # note our Envpool + PPO only gets 0 as the base return
    "BattleZone-v5": (2360.0, 37187.5),
    "BeamRider-v5": (363.9, 16926.5),
    "Berzerk-v5": (123.7, 2630.4),
    "Bowling-v5": (23.1, 160.7),
    "Boxing-v5": (0.1, 12.1),
    "Breakout-v5": (1.7, 30.5),
    "Centipede-v5": (2090.9, 12017.0),
    "ChopperCommand-v5": (811.0, 7387.8),
    "CrazyClimber-v5": (10780.5, 35829.4),
    "Defender-v5": (2874.5, 18688.9),  ## TODO: where is defender in the original DQN paper?
    "DemonAttack-v5": (152.1, 1971.0),
    "DoubleDunk-v5": (-18.6, -16.4),
    "Enduro-v5": (0.0, 860.5),
    "FishingDerby-v5": (-91.7, -38.7),
    "Freeway-v5": (0.0, 29.6),
    "Frostbite-v5": (65.2, 4334.7),
    "Gopher-v5": (257.6, 2412.5),
    "Gravitar-v5": (173.0, 3351.4),
    "Hero-v5": (1027.0, 30826.4),
    "IceHockey-v5": (-11.2, 0.9),
    "Jamesbond-v5": (29.0, 302.8),
    "Kangaroo-v5": (52.0, 3035.0),
    "Krull-v5": (1598.0, 2665.5),
    "KungFuMaster-v5": (258.5, 22736.3),
    "MontezumaRevenge-v5": (0.0, 4753.3),
    "MsPacman-v5": (307.3, 6951.6),
    "NameThisGame-v5": (2292.3, 8049.0),
    "Phoenix-v5": (761.4, 7242.6),  ## TODO: where is Phoenix in the original DQN paper?
    "Pitfall-v5": (-229.4, 6463.7),  ## TODO: where is Pitfall in the original DQN paper?
    "Pong-v5": (-20.7, 14.6),
    "PrivateEye-v5": (24.9, 69571.3),
    "Qbert-v5": (163.9, 13455.0),
    "Riverraid-v5": (1338.5, 17118.0),
    "RoadRunner-v5": (11.5, 7845.0),
    "Robotank-v5": (2.2, 11.9),
    "Seaquest-v5": (68.4, 42054.7),
    "Skiing-v5": (
        -17098.1,
        -4336.9,
    ),  # note our Envpool + PPO only gets -28500 as the base return ## TODO: where is Skiing in the original DQN paper?
    "Solaris-v5": (1236.3, 12326.7),  ## TODO: where is Solaris in the original DQN paper?
    "SpaceInvaders-v5": (148.0, 1668.7),
    "StarGunner-v5": (664.0, 10250.0),
    "Surround-v5": (-10.0, 6.5),  ## TODO: where is Surround in the original DQN paper?
    "Tennis-v5": (-23.8, -8.3),
    "TimePilot-v5": (3568.0, 5229.2),
    "Tutankham-v5": (11.4, 167.6),
    "UpNDown-v5": (533.4, 11693.2),
    "Venture-v5": (0.0, 1187.5),
    "VideoPinball-v5": (16256.9, 17667.9),
    "WizardOfWor-v5": (563.5, 4756.5),  # note our Envpool + PPO only gets 0 as the base return
    "YarsRevenge-v5": (3092.9, 54576.9),  ## TODO: where is YarsRevenge in the original DQN paper?
    "Zaxxon-v5": (32.5, 9173.3),
}
if __name__ == "__main__":
    args = parse_args()
    runset_names = set()
    for file in args.files:
        df = pd.read_csv(file, index_col=0)
        # env_ids = atari_human_normalized_scores.keys()
        env_ids = df.index
        for runset_name in df.columns:
            if runset_name in runset_names:
                continue
            runset_names.add(runset_name)
            hnss = []
            for env_id in env_ids:
                # if env_id not in df.index:
                #     continue
                episodic_return = float(df.loc[env_id, runset_name].split(" ± ")[0])
                hns = (episodic_return - atari_human_normalized_scores[env_id][0]) / (
                    atari_human_normalized_scores[env_id][1] - atari_human_normalized_scores[env_id][0]
                )
                hnss += [hns]
            print(f"{runset_name}")
            print(f"┣━━ median hns: {np.median(hnss)}")
            print(f"┣━━ mean hns: {np.mean(hnss)}")
