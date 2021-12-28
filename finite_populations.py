import numpy as np
from game import HDG
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from typing import Tuple


class FiniteNPlayerHDGDynamics:

    def __init__(self, Z: int, N: int, c_h: float, w: float):
        """

        Parameters
        ----------
        Z: int
            Population size.
        N: int
            Sample size.
        c_h: float
            Cost of injury for hawks.
        w:
            intensity of selection (sometimes called beta)
        """
        self.Z = Z
        self.N = N
        self.c_h = c_h
        self.w = w

    def compute_fitnesses(self, k: int) -> Tuple[float, float]:
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
            hdg_hawk = HDG(self.N, i, self.c_h)
            hdg_dove = HDG(self.N, i+i, self.c_h)

            payoff_hawk,_ = hdg_hawk.expected_payoffs()
            _, payoff_dove = hdg_dove.expected_payoffs()

            fh += hypergeom(self.Z-1, k, self.N-1).pmf(i) * payoff_hawk
            fd += hypergeom(self.Z-1, k-1, self.N-1).pmf(i) * payoff_dove

        return fh, fd

    def gradient_selection(self, k: int) -> float:
        """
        Compute the gradient of selection for given value of k.

        Parameters
        ----------
        k: int
            Number of doves in population.

        Returns
        -------
            gradient of selection for given k.
        """

        fh, fd = self.compute_fitnesses(k)

        weighted_diff = self.w*(fd - fh)
        pop_factor = (k*(self.Z - k)) / (self.Z**2)

        T_plus = pop_factor / (1 + np.exp(-weighted_diff))
        T_minus = pop_factor / (1 + np.exp(weighted_diff))

        return T_plus - T_minus

    def plot_gradient_selection(self):
        """
        Plot the gradient of selection as function of the fraction k/Z.
        """
        ks = np.arange(self.Z+1)

        gradient = [self.gradient_selection(k) for k in ks]

        plt.figure()
        plt.plot(ks/self.Z, gradient)
        plt.xlabel("k/Z")
        plt.ylabel("G(k)")
        plt.show()


if __name__ == "__main__":
    finite = FiniteNPlayerHDGDynamics(100, 5, 0.5, 1)
    finite.plot_gradient_selection()



