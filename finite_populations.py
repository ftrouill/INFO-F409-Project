import numpy as np
from game import HDG
import matplotlib.pyplot as plt
from typing import Tuple

from dataclasses import dataclass


@dataclass
class FiniteNPlayerHDGDynamics:
    """

    Parameters
    ----------
    Z: int
        Population size.
    N: int
        Sample size.
    w:
        intensity of selection (sometimes called beta)
    """

    Z: int
    N: int
    w: float

    def gradient_selection(self, k: int, c_h: float) -> float:
        """
        Compute the gradient of selection for given value of k.

        Parameters
        ----------
        k: int
            Number of doves in population.
        c_h: float
            Cost of injury for hawks.

        Returns
        -------
            gradient of selection for given k.
        """

        fh, fd = HDG(self.N, c_h).average_fitness_finite_pop(k, self.Z)

        weighted_diff = self.w * (fd - fh)
        pop_factor = (k * (self.Z - k)) / (self.Z ** 2)

        T_plus = pop_factor / (1 + np.exp(-weighted_diff))
        T_minus = pop_factor / (1 + np.exp(weighted_diff))

        return T_plus - T_minus

    def compute_full_gradient(self, c_h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the gradient of selection as a function of k.
        Parameters
        ----------
        c_h: float
            Cost of injury for hawks.

        Returns
        -------
            tuple(k_values, gradient) where each element of gradient is the gradient of selection for the
        corresponding value of k.
        """
        k_values = np.arange(self.Z + 1)

        gradient = np.array([self.gradient_selection(k, c_h) for k in k_values])

        return k_values, gradient

    def plot_gradient_selection(self, c_h: float) -> plt.Figure:
        """
        Plot the gradient of selection as function of the fraction k/Z.
        Parameters
        ----------
        c_h: float
            Cost of injury for hawks.
        Returns
        -------
            fig: The figure containing the plot of the gradient.
        """

        k_values, gradient = self.compute_full_gradient(c_h)

        fig = plt.figure()
        plt.plot(k_values / self.Z, gradient)
        plt.xlabel("k/Z")
        plt.ylabel("G(k)")

        return fig

    @staticmethod
    def find_equilibrium(gradient: np.ndarray):
        """
        Find the equilibrium value of the gradient of selection, i.e. its zero.
        Parameters
        ----------
        gradient: np.ndarray
            Array containing values of the gradient as a function of k.

        Returns
        -------
            Value of k where the gradient is zero, or NaN
        """
        gradient = gradient[1:-1]

        equilibrium = np.argmin(np.abs(gradient))

        return equilibrium if equilibrium != 0 else np.nan


def plot_equilibria(Z: int, w: float):
    """
    PLot the value of the equilibrium points of the gradient as a function of C_H, for different values of N.
    Parameters
    ----------
    Z: int
        Population size
    w: float
        intensity of selection (sometimes called beta)

    Returns
    -------

    """
    ch_values = np.arange(0.0, 1, 0.01)

    N_values = [5, 10, 20, 50, 100]

    plt.figure()
    for N in N_values:

        finite = FiniteNPlayerHDGDynamics(Z, N, w)
        equilibria = np.array([finite.find_equilibrium(finite.compute_full_gradient(c_h)[1]) for c_h in ch_values])
        plt.plot(ch_values, equilibria/Z, label=N)

    plt.xlabel("$C_H$")
    plt.ylabel("$k^*/Z$")
    plt.legend()
    #plt.savefig("plots/equilibria.png")
    plt.show()


if __name__ == "__main__":
    """finite = FiniteNPlayerHDGDynamics(100, 5, 1)
    fig, gradient = finite.plot_gradient_selection(0.5)
    plt.show()
    gradient[40] = 0.0
    print(gradient[39:44])
    print(finite.find_equilibrium(gradient))"""
    plot_equilibria(100, 1)


