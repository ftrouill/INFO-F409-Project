class HDG:
    def __init__(self, N: int, n_doves: int, c_h, R: float = 1.0):
        """
        Creates an N-player Hawk-Dove Game. Parameter R is assumed to be 1.
        :param N: number of players
        :param n_doves: number of doves
        :param c_h: cost
        :param R: resource (=1.0 by default)
        """
        self.N = N
        self.n_hawks = N - n_doves
        self.n_doves = n_doves
        self.c_h = c_h
        self.R = R

    def expected_payoffs(self) -> (float, float):
        """
        Calculates the expected payoff for the game.
        :return: tuple (hawk_reward, dove_reward)
        """
        if self.n_hawks == 0:
            return 0.0, self.R / self.N  # change hawks payoffs to None instead of 0?
        else:
            return (self.R - (self.n_hawks - 1) * self.c_h) / self.n_hawks, 0.0


if __name__ == '__main__':
    # petits tests pour verifier que ça fonctionne
    print(HDG(30, 30, 0.5).expected_payoffs())
    print(HDG(30, 29, 0.5).expected_payoffs())
    print(HDG(30, 28, 0.5).expected_payoffs())
    print(HDG(30, 25, 0.5).expected_payoffs())
