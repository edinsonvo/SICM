from dataclasses import dataclass
from datetime import datetime
from copy import deepcopy

@dataclass
class Scenario:

    name: str

    config: object

    results: object = None

    created_at: str = datetime.now().isoformat()


class ScenarioManager:

    def __init__(self):

        self.scenarios = {}

    def save(self, scenario):

        self.scenarios[scenario.name] = scenario

    def load(self, name):

        return self.scenarios.get(name)

    def list_scenarios(self):

        return list(self.scenarios.keys())

    def compare(self, base_name, compare_name):

        base = self.scenarios[base_name]

        comp = self.scenarios[compare_name]

        return {

            "Y":

                comp.results.Y
                - base.results.Y,

            "r":

                comp.results.r
                - base.results.r,

            "inflation":

                comp.results.inflation
                - base.results.inflation,

            "employment":

                comp.results.employment
                - base.results.employment
        }

    def clone(self, name, new_name):

        scenario = deepcopy(self.scenarios[name])

        scenario.name = new_name

        self.save(scenario)

        return scenario
