import hashlib
import json


class SimulationCache:

    def __init__(self):

        self.cache = {}

    def make_key(
        self,
        model_name,
        config
    ):

        payload = {

            "model":
                model_name,

            "config":
                config.to_dict()
        }

        text = json.dumps(
            payload,
            sort_keys=True
        )

        return hashlib.md5(
            text.encode()
        ).hexdigest()

    def get(
        self,
        key
    ):

        return self.cache.get(key)

    def save(
        self,
        key,
        result
    ):

        self.cache[key] = result
