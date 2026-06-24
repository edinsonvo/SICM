class EconomicValidator:

    @staticmethod
    def validate_consumption(c):

        if c <= 0 or c >= 1:

            raise ValueError(
                "c debe estar entre 0 y 1"
            )

    @staticmethod
    def validate_money_supply(M):

        if M <= 0:

            raise ValueError(
                "M debe ser positiva"
            )

    @staticmethod
    def validate_prices(P):

        if P <= 0:

            raise ValueError(
                "P debe ser positiva"
            )
