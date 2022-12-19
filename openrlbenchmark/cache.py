import os
import wandb
import json
import pandas as pd

class CachedRun:
    def __init__(self, run: wandb.apis.public.Run, cache_dir: str = None):
        self.run = run

        if cache_dir is not None:
            self.dir = os.path.join(cache_dir, *run.path)
            self.run_df_path = os.path.join(self.dir, "run_df.csv")
            self.run_path = os.path.join(self.dir, "run.json")

            print(os.path.exists(self.run_path))
            if not os.path.exists(self.run_path):
                os.makedirs(self.dir, exist_ok=True)
                self.run_df = pd.DataFrame([row for row in run.scan_history()])
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
                with open(self.run_path, "r") as f:
                    self.run = json.load(f)
                self.run_df = pd.read_csv(self.run_df_path)
        

