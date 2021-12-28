from typing import Tuple
import numpy as np

import unittest


class HDG:
    def __init__(self, N: int, n_doves: int, c_h: float, R: float = 1.0):
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

    def expected_payoffs(self) -> Tuple[float]:
        """
        Calculates the expected payoff for the game.
        :return: tuple (hawk_reward, dove_reward)
        """
        if self.n_hawks == 0:
            return 0.0, self.R / self.N  # change hawks payoffs to None instead of 0?
        else:
            return (self.R - (self.n_hawks - 1) * self.c_h) / self.n_hawks, 0.0

    def payoff_matrix(self) -> np.ndarray:
        """Computes the payoff matrix from current or given state.

        :param state: State (numbre of hawks) to replace this instance's state, defaults to None
        :type state: int, optional
        :return: Payoff matrix
        :rtype: np.ndarray
        """
        payoffs = self.expected_payoffs()
        return np.repeat(payoffs, 2).reshape((2, 2))


class HDGTest(unittest.TestCase):
    def test_payoffs(self):
        hdg = HDG(30, 30, 0.5)
        self.assertEqual(hdg.expected_payoffs(), (0.0, 1 / 30))
        hdg = HDG(30, 29, 0.5)
        self.assertEqual(hdg.expected_payoffs(), (1, 0.0))
        hdg = HDG(30, 28, 0.5)
        self.assertEqual(hdg.expected_payoffs(), (0.5 / 2, 0.0))
        hdg = HDG(30, 25, 0.5)
        self.assertEqual(hdg.expected_payoffs(), (-1 / 5, 0.0))


class HDG_T(HDG):
    """
    Creates an N-player Hawk-Dove Game with threshold above which doves can aggregate and protect the resource.
    Parameter R is assumed to be 1.
    :param N: number of players
    :param n_doves: number of doves
    :param c_h: fighting cost for hawks
    :param c_d: protecting cost for doves
    :param T: threshold for which dove are sufficiently many to protect resource from hawks
    :param R: resource (=1.0 by default)
    """

    def __init__(
        self, N: int, n_doves: int, c_h: float, c_d: float, T: float, R: float = 1.0
    ):
        super().__init__(N, n_doves, c_h, R)
        self.c_d = c_d
        self.T = T

    def expected_payoffs(self) -> Tuple[float]:
        if self.n_doves / self.N >= self.T:
            return 0.0, (self.R - self.n_hawks * self.c_d) / self.n_doves
        else:
            return super().expected_payoffs()


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
