from math import comb
from typing import Tuple
import numpy as np
from scipy.stats import hypergeom, binom
from time import perf_counter

import unittest

from dataclasses import dataclass


@dataclass
class HDG:
    """
    Represents an N-player Hawk-Dove Game. Parameter R is assumed to be 1.
    :param N: number of players
    :param c_h: cost
    :param R: resource (=1.0 by default)
    """

    N: int
    c_h: float
    R: float = 1.0

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
        b = binom.pmf(range(self.N), self.N-1, x)
        f_h = sum(
            [
                b[i]
                * self.expected_payoffs(i)[0]
                for i in range(self.N)
            ]
        )
        f_d = sum(
            [
                b[i]
                * self.expected_payoffs(i + 1)[1]
                for i in range(self.N)
            ]
        )
        return f_h, f_d

    def average_fitness_finite_pop(self, k: int, Z: int) -> Tuple[float, float]:
        """
        Compute the average fitness of the two strategies in the finite population.

        Parameters
        ----------
        k: int
            Number of doves in the population.
        z: int
            Size of the finite population.

        Returns
        -------
            tuple(fitness_hawk, fitness_dove)
        """
        fd = 0
        fh = 0
        hyp_h = hypergeom.pmf(range(self.N), Z - 1, k, self.N - 1)
        hyp_d = hypergeom.pmf(range(self.N), Z - 1, k - 1, self.N - 1)
        for i in range(self.N):
            payoff_hawk, _ = self.expected_payoffs(i)
            _, payoff_dove = self.expected_payoffs(i + 1)

            fh += hyp_h[i] * payoff_hawk
            fd += hyp_d[i] * payoff_dove

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


@dataclass
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

    c_d: float = 0.5
    T: float = 0.5

    def expected_payoffs(self, n_doves) -> Tuple[float, float]:
        n_hawks = self.N - n_doves
        if n_doves / self.N >= self.T:
            return 0.0, (self.R - n_hawks * self.c_d) / n_doves
        else:
            return super().expected_payoffs(n_doves)


class HGDTTest(unittest.TestCase):
    def test_payoffs(self):
        hdgt = HDG_T(N=30, c_h=0.5, c_d=0.2, T=0.2)
        self.assertEqual(hdgt.expected_payoffs(30), (0.0, 1 / 30))
        self.assertEqual(hdgt.expected_payoffs(10), (0.0, -3 / 10))
        self.assertEqual(hdgt.expected_payoffs(0), (-13.5 / 30, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
    # hdg = HDG(30, 28)
    # now = perf_counter()
    # for i in range(100):
    #     hdg.average_fitness_infinite_pop(i/100)
    # print(f"infinite: {perf_counter() - now}")
    # now = perf_counter()
    # for i in range(100):
    #     hdg.average_fitness_finite_pop(i, 1000)
    # print(f"finite: {perf_counter() - now}")