from game import HDG

from egttools.utils import find_saddle_type_and_gradient_direction
from egttools.plotting import plot_gradient

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


class InfiniteNPlayerHDGDynamics:
    def __init__(
        self,
        N: int,
        c_h: float,
        R: float = 1.0,
        nb_states: int = 101,
        nb_costs: int = 100,
    ):
        """Generates replicator dynamics for infinite population.

        Parameters
        ----------
        N : int
            Sample size.
        c_h : float
            Cost of injury for hawks.
        R : float
            Resource reward.

        """
        self.N = N
        self.c_h = c_h
        self.R = R
        self.nb_states = nb_states
        self.nb_costs = nb_costs

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

    def plot_gradient(self, **kwargs) -> plt.Axes:
        """Generates the plot of the gradient.

        Returns
        -------
        plt.Axes
            Matplotlib axes object.
        """
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
        ax = plot_gradient(
            dove_strategy,
            G,
            saddle_points,
            saddle_type,
            gradient_direction,
            "N-player Hawk-Dove game replicator dynamics",
            xlabel="$x$",
            label=self.c_h,
            **kwargs,
        )
        ax.set_label(self.c_h)
        return ax

    def compute_equilibria_cost(self):
        # array of different costs
        costs = np.linspace(0, 1, num=self.nb_costs, dtype=np.float64)
        # array of states
        dove_strategy = np.linspace(0, 1, num=self.nb_states, dtype=np.float64)
        # find optimal state for each cost
        equilibria = np.zeros_like(costs)
        for i, c in enumerate(costs):
            # update cost
            self.c_h = c
            # compute gradient for each state
            G = np.array([self.compute_gradient_for_state(i) for i in dove_strategy])
            # remove first and last point
            G = G[1:-1]
            # get the equilibria
            equilibria[i] = np.argmin(np.abs(G)) / self.nb_states
        return equilibria


def plot_hdg_gradient(N=5):
    costs = [0.1, 0.5, 0.9]
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    patches = np.zeros_like(costs, dtype=object)
    for i, c in enumerate(costs):
        rep_dyn = InfiniteNPlayerHDGDynamics(N, c, nb_states=10000)
        rep_dyn.plot_gradient(ax=ax)
        patches[i] = mpatches.Patch(
            color=list(mcolors.TABLEAU_COLORS.values())[i], label=c
        )
    ax.legend(handles=patches.tolist())
    plt.show()


def plot_equilibria():
    N_values = [5, 10, 20, 50, 100]
    for N in N_values:
        rep_dyn = InfiniteNPlayerHDGDynamics(N, 0.9, nb_states=10000)
        eq = rep_dyn.compute_equilibria_cost(epsilon=1e-6)
        plt.plot(np.linspace(0, 1, rep_dyn.nb_costs), eq, label=N)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_hdg_gradient()
    plot_equilibria()
