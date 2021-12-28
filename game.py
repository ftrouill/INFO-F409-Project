from typing import Tuple
from math import comb
import numpy as np
from scipy.stats import hypergeom

import unittest


class HDG:
    def __init__(self, N: int, c_h: float, R: float = 1.0):
        """
        Creates an N-player Hawk-Dove Game. Parameter R is assumed to be 1.
        :param N: number of players
        :param n_doves: number of doves
        :param c_h: cost
        :param R: resource (=1.0 by default)
        """
        self.N = N
        self.c_h = c_h
        self.R = R

    def expected_payoffs(self, n_doves) -> Tuple[float, float]:
        """
        Calculates the expected payoff for the game.
        :return: tuple (hawk_reward, dove_reward)
        """
        n_hawks = self.N - n_doves
        if n_hawks == 0:
            return 0.0, self.R / self.N  # change hawks payoffs to None instead of 0?
        else:
            return (self.R - (n_hawks - 1) * self.c_h) / n_hawks, 0.0

    def average_fitness_infinite_pop(self, x: float) -> Tuple[float, float]:
        payoffs = [self.expected_payoffs(i) for i in range(self.N+1)]
        f_h = sum([comb(self.N-1, i)*(x**i)*((1-x)**(self.N-1-i))*payoffs[i][0] for i in range(self.N)])
        f_d = sum([comb(self.N-1, i)*(x**i)*((1-x)**(self.N-1-i))*payoffs[i+1][1] for i in range(self.N)])
        return f_h, f_d

    def average_fitness_finite_pop(self, k: int, Z: int) -> Tuple[float, float]:
        """
        Compute the average fitness of the two strategies in the finite population.

        Parameters
        ----------
        k: int
            Number of doves in the population.

        Returns
        -------
            tuple(fitness_hawk, fitness_dove)
        """
        fd = 0
        fh = 0
        for i in range(self.N):
            payoff_hawk, _ = self.expected_payoffs(i)
            _, payoff_dove = self.expected_payoffs(i+1)

            fh += hypergeom(Z-1, k, self.N-1).pmf(i) * payoff_hawk
            fd += hypergeom(Z-1, k-1, self.N-1).pmf(i) * payoff_dove

        return fh, fd


class HDGTest(unittest.TestCase):
    def test_payoffs(self):
        hdg = HDG(30, 0.5)
        self.assertEqual(hdg.expected_payoffs(30), (0.0, 1 / 30))
        hdg = HDG(30, 0.5)
        self.assertEqual(hdg.expected_payoffs(29), (1, 0.0))
        hdg = HDG(30, 0.5)
        self.assertEqual(hdg.expected_payoffs(28), (0.5 / 2, 0.0))
        hdg = HDG(30, 0.5)
        self.assertEqual(hdg.expected_payoffs(25), (-1 / 5, 0.0))


class HDG_T(HDG):
    """
    Creates an N-player Hawk-Dove Game with threshold above which doves can aggregate and protect the resource.
    Parameter R is assumed to be 1.
    :param N: number of players
    :param c_h: fighting cost for hawks
    :param c_d: protecting cost for doves
    :param T: threshold for which dove are sufficiently many to protect resource from hawks
    :param R: resource (=1.0 by default)
    """

    def __init__(
        self, N: int, c_h: float, c_d: float, T: float, R: float = 1.0
    ):
        super().__init__(N, c_h, R)
        self.c_d = c_d
        self.T = T

    def expected_payoffs(self, n_doves) -> Tuple[float, float]:
        n_hawks = self.N - n_doves
        if n_doves / self.N >= self.T:
            return 0.0, (self.R - n_hawks * self.c_d) / n_doves
        else:
            return super().expected_payoffs(n_doves)


if __name__ == "__main__":
    # petits tests pour verifier que ça fonctionne
    # print(HDG(30, 30, 0.5).expected_payoffs())
    # print(HDG(30, 29, 0.5).expected_payoffs())
    # print(HDG(30, 28, 0.5).expected_payoffs())
    # print(HDG(30, 25, 0.5).expected_payoffs())
    # print(HDG_T(30, 28, 0.5, 0.2, 0.2).expected_payoffs())
    # print(HDG_T(30, 28, 0.5, 0.5, 0.2).expected_payoffs())
    # print(HDG_T(30, 28, 0.5, 0.9, 0.2).expected_payoffs())
    unittest.main()
