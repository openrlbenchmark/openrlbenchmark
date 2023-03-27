import json
import os

import pandas as pd
import tqdm
import wandb


class CachedRun:
    def __init__(self, run: wandb.apis.public.Run, cache_dir: str = None):
        self.run = run

        if cache_dir is not None:
            self.dir = os.path.join(cache_dir, *run.path)
            self.run_df_path = os.path.join(self.dir, "run_df.csv")
            self.run_path = os.path.join(self.dir, "run.json")

            # print(os.path.exists(self.run_path))
            if not os.path.exists(self.run_path):
                os.makedirs(self.dir, exist_ok=True)
                rows = []
                for row in tqdm.tqdm(run.scan_history()):
                    rows.append(row)
                self.run_df = pd.DataFrame(rows)
                if "videos" in self.run_df:
                    self.run_df = self.run_df.drop(columns=["videos"], axis=1)
                self.run_df.to_csv(self.run_df_path)

                # remove unpickable objects
                del run.client
                del run.user._client
                run.user = run.user.__dict__
                with open(self.run_path, "w") as f:
                    json.dump(self.run.__dict__, f)
            else:
                with open(self.run_path) as f:
                    self.run = json.load(f)
                self.run_df = pd.read_csv(self.run_df_path)
