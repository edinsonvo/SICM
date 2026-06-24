import time


class Benchmark:

    @staticmethod
    def measure(
        func,
        *args,
        **kwargs
    ):

        start = time.perf_counter()

        result = func(
            *args,
            **kwargs
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        return result, elapsed
