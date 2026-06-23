from core.shocks import (
    FiscalShock,
    MonetaryShock,
    SupplyShock,
    ExternalShock
)


class EconomicInterpreter:

    def explain(self, shock):

        if isinstance(shock, FiscalShock):

            return """
            Choque Fiscal:

            • Aumenta la demanda agregada.

            • IS se desplaza a la derecha.

            • Aumenta el ingreso.

            • Tiende a aumentar la tasa de interés.
            """

        elif isinstance(shock, MonetaryShock):

            return """
            Choque Monetario:

            • LM se desplaza a la derecha.

            • Disminuye la tasa de interés.

            • Aumenta inversión y producción.
            """

        elif isinstance(shock, SupplyShock):

            return """
            Choque de Oferta:

            • Aumenta capacidad productiva.

            • Desplaza OA.

            • Incrementa producto potencial.
            """

        elif isinstance(shock, ExternalShock):

            return """
            Choque Externo:

            • Modifica exportaciones netas.

            • Afecta balanza de pagos.

            • Cambia equilibrio externo.
            """

        return "Sin interpretación disponible."
