import numpy as np
from game import HDG
import matplotlib.pyplot as plt
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

        fh, fd = HDG(self.N, self.c_h).average_fitness_finite_pop(k, self.Z)

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



