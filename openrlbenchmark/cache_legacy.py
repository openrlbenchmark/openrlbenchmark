import json
import os

import pandas as pd
import tqdm
import wandb
from dotmap import DotMap


class CachedRun:
    def __init__(self, run: wandb.apis.public.Run, cache_dir: str = None):
        self.run = run
        if cache_dir is not None:
            self.dir = os.path.join(cache_dir, *[run.entity, run.project, run.id])
            self.run_df_path = os.path.join(self.dir, "run_df.csv")
            self.run_path = os.path.join(self.dir, "run.json")
            if not os.path.exists(self.run_path):
                os.makedirs(self.dir, exist_ok=True)
                rows = []
                for row in tqdm.tqdm(run.scan_history()):
                    rows.append(row)
                self.run_df = pd.DataFrame(rows)
                if "videos" in self.run_df:
                    self.run_df = self.run_df.drop(columns=["videos"], axis=1)
                self.run_df.to_csv(self.run_df_path)
                self.dump_json(run)
            else:
                with open(self.run_path) as f:
                    self.run = json.load(f)
                if "_attrs" in self.run:  # legacy format, create the new format
                    self.dump_json(run)
                    with open(self.run_path) as f:
                        self.run = json.load(f)
                self.run = DotMap(self.run)  # give dot access
                self.run_df = pd.read_csv(self.run_df_path)

    def dump_json(self, run):
        run_attr = {
            attr: getattr(run, attr) for attr in ["id", "name", "state", "url", "project", "entity", "config", "tags", "path"]
        }
        run_attr["user"] = {"username": run.user.username, "name": run.user.name}
        with open(self.run_path, "w") as f:
            json.dump(run_attr, f)
