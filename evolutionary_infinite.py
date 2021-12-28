from game import HDG

from egttools.utils import find_saddle_type_and_gradient_direction
from egttools.plotting import plot_gradient

import numpy as np
import matplotlib.pyplot as plt


class InfiniteNPlayerHDGDynamics:
    def __init__(self, N: int, c_h: float, R: float = 1.0, nb_states: int = 101):
        """Generates replicator dynamics for infinite population.

        Parameters
        ----------
        N : int
            Sample size.
        c_h : float
            Cost of injury for hawks.
        """
        self.N = N
        self.c_h = c_h
        self.R = R
        self.nb_states = nb_states

    def compute_gradient_for_state(self, x: float) -> float:
        """Compute the gradient for a given state.

        Parameters
        ----------
        x : float
            Fraction of doves in the population.
        """
        # we use the direct analytical formula
        # developed in the paper
        return (x / self.N) * (
            self.c_h * np.power(x, self.N)
            + self.R * np.power(x, self.N - 1)
            - self.N * self.c_h * x
            + self.N * self.c_h
            - self.R
            - self.c_h
        )

    def plot_gradient(self):
        # generate array of states
        dove_strategy = np.linspace(0, 1, num=self.nb_states, dtype=np.float64)
        # compute gradient
        G = np.array([self.compute_gradient_for_state(i) for i in dove_strategy])
        # find saddle points
        epsilon = 1e-5
        saddle_points_idx = np.where((G <= epsilon) & (G >= -epsilon))[0]
        saddle_points = saddle_points_idx / (self.nb_states - 1)
        saddle_type, gradient_direction = find_saddle_type_and_gradient_direction(
            G, saddle_points_idx
        )
        return plot_gradient(
            dove_strategy,
            G,
            saddle_points,
            saddle_type,
            gradient_direction,
            "N-player Hawk-Dove game replicator dynamics",
            xlabel="$x$",
        )


if __name__ == "__main__":
    rep_dyn = InfiniteNPlayerHDGDynamics(5, 0.9, nb_states=10000)
    ax = rep_dyn.plot_gradient()
    plt.show()
