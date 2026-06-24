import json
from pathlib import Path


class ScenarioStorage:

    def __init__(self, storage_dir="scenarios"):

        self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(self, scenario):

        filename = (
            self.storage_dir
            / f"{scenario.name}.json"
        )

        data = {

            "name": scenario.name,

            "created_at":
                scenario.created_at,

            "config":
                scenario.config.to_dict(),

            "results":
                scenario.results.__dict__
                if scenario.results
                else None
        }

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4
            )

    def load(self, filename):

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)
