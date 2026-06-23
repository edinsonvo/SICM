from solvers.islm_solver import ISLMSolver

from solvers.mundell_fleming_solver import (
    MundellFlemingSolver
)

from solvers.classical_closed_solver import (
    ClassicalClosedSolver
)

from solvers.classical_open_solver import (
    ClassicalOpenSolver
)


class ModelFactory:

    @staticmethod
    def get_solver(model_name):

        models = {

            "islm":

                ISLMSolver(),

            "mundell_fleming":

                MundellFlemingSolver(),

            "classical_closed":

                ClassicalClosedSolver(),

            "classical_open":

                ClassicalOpenSolver()
        }

        return models[model_name]
