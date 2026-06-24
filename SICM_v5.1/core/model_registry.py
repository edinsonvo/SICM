AVAILABLE_MODELS = {

    "islm":

        "Keynesiano Cerrado",

    "mundell_fleming":

        "Keynesiano Abierto",

    "classical_closed":

        "Clásico Cerrado",

    "classical_open":

        "Clásico Abierto"
}
class ModelRegistry:

    MODELS = {

        "islm": {

            "name":
                "Keynesiano Cerrado",

            "solver":
                "ISLMSolver"
        },

        "mundell_fleming": {

            "name":
                "Keynesiano Abierto",

            "solver":
                "MundellFlemingSolver"
        },

        "classical_closed": {

            "name":
                "Clásico Cerrado",

            "solver":
                "ClassicalClosedSolver"
        },

        "classical_open": {

            "name":
                "Clásico Abierto",

            "solver":
                "ClassicalOpenSolver"
        }
    }
