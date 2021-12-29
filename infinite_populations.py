from game import HDG_T

from egttools.utils import find_saddle_type_and_gradient_direction
from egttools.plotting import plot_gradient as egt_gradient

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from dataclasses import dataclass

from typing import Tuple, Dict, List


@dataclass
class InfiniteNPlayerHDGDynamics:
    """Generates replicator dynamics for HDG with infinite population.

    Parameters
    ----------
    N : int
        Sample size.
    c_h : float
        Cost of injury for hawks.
    R : float
        Resource reward.
    nb_states : int
        State resolution for plotting.
    nb_costs : int
        Costs resolution for plotting.
    """

    N: int
    c_h: float
    R: float = 1.0
    nb_states: int = 101
    nb_costs: int = 100

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
        # G2 = np.array([x * (1 - x) *
        #                (HDG(self.N, self.c_h, self.R).average_fitness_infinite_pop(x)[1] -
        #                 HDG(self.N, self.c_h, self.R).average_fitness_infinite_pop(x)[0])
        #                for x in dove_strategy])
        # print(G)
        # print(G2)
        # print(G2 - G)
        # find saddle points
        epsilon = 1e-6
        saddle_points_idx = np.where((np.roll(G, -1) * G < 0) | (np.abs(G) <= epsilon))[
            0
        ]
        saddle_points = saddle_points_idx / (self.nb_states - 1)
        saddle_type, gradient_direction = find_saddle_type_and_gradient_direction(
            G, saddle_points_idx
        )
        ax = egt_gradient(
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

    def compute_hdg_equilibria_cost_c_h(self) -> List:
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
            epsilon = 1e-6
            saddle_points_idx = np.where(
                (np.roll(G, -1) * G < 0) | (np.abs(G) <= epsilon)
            )[0]
            saddle_points = saddle_points_idx / (self.nb_states - 1)
            if saddle_points.size > 0:
                equilibria[i] = saddle_points[0]
        return equilibria

    @staticmethod
    def plot_hdg_gradient(N=5) -> None:
        """Plots replicator gradient for different cost values.

        Parameters
        ----------
        N : int, optional
            Sample size, by default 5
        """
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

    @staticmethod
    def plot_hdg_equilibria() -> None:
        """Plots optimal state for different cost values."""
        N_values = [5, 10, 20, 50, 100]
        for N in N_values:
            rep_dyn = InfiniteNPlayerHDGDynamics(N, 0.9, nb_states=10000)
            eq = rep_dyn.compute_hdg_equilibria_cost_c_h()
            plt.plot(np.linspace(0, 1, rep_dyn.nb_costs), eq, label=N)
        plt.legend()
        plt.show()


@dataclass
class InfiniteNPlayerHDGTDynamics(InfiniteNPlayerHDGDynamics):
    """Generates replicator dynamics for HDGT with infinite population.

    Parameters
    ----------
    c_d : float
        Cost for doves to protect the resource.
    T : float
        Doves threshold to protect the resource.
    """

    c_d: float = 0.5
    T: float = 0.5

    def compute_gradient_for_state(self, x: float) -> float:
        hdgt = HDG_T(self.N, self.c_h, self.R, self.c_d, self.T)
        # compute fitnesses
        f_h, f_d = hdgt.average_fitness_infinite_pop(x)
        # compute gradient
        return x * (1 - x) * (f_d - f_h)

    @staticmethod
    def plot_c_h_equilibria(N=5, T=0.2):
        c_d_values = [0.2, 0.5, 0.8]
        colors = ["red", "blue", "black"]
        for c_d, color in zip(c_d_values, colors):
            print(c_d)
            rep_dyn = InfiniteNPlayerHDGTDynamics(N, 0.9, nb_states=10000, T=T, c_d=c_d)
            un_eq, st_eq = rep_dyn.compute_hdgt_equilibria_cost_c_h()
            plt.plot(
                st_eq.keys(),
                st_eq.values(),
                label=N,
                color=color,
            )
            plt.plot(
                un_eq.keys(),
                un_eq.values(),
                "--",
                label=N,
                color=color,
            )
        plt.legend()
        plt.show()

    def compute_hdgt_equilibria_cost_c_h(self) -> Tuple[Dict, Dict]:
        # array of different costs
        costs = np.linspace(0, 1, num=self.nb_costs, dtype=np.float64)
        # array of states
        dove_strategy = np.linspace(0, 1, num=self.nb_states, dtype=np.float64)
        # find optimal state for each cost
        unstable_equilibria = dict()
        stable_equilibria = dict()
        for i, c in enumerate(costs):
            # update cost
            self.c_h = c
            # compute gradient for each state
            G = np.array([self.compute_gradient_for_state(i) for i in dove_strategy])
            epsilon = 1e-6
            saddle_points_idx = np.where(
                (np.roll(G, -1) * G < 0) | (np.abs(G) <= epsilon)
            )[0]
            saddle_points = saddle_points_idx / (self.nb_states - 1)
            # remove points too close to the beginning or the end
            # saddle_points = np.delete(
            #     saddle_points,
            #     np.where((saddle_points <= 1e-2) | (saddle_points >= 1 - 1e-2)),
            # )
            print(c)
            print(saddle_points)
            if saddle_points.size > 2:
                # TODO split stable and unstable
                saddle_types = find_saddle_type_and_gradient_direction(
                    G, saddle_points_idx
                )[0]
                # remove first and last equilibria
                saddle_points = saddle_points[1:-1]
                saddle_types = saddle_types[1:-1]
                # use min() to remove duplicates
                try:
                    unstable_equilibria[c] = saddle_points[~saddle_types].max()
                except ValueError:
                    # there is no unstable equilibrium; pass
                    pass
                try:
                    stable_equilibria[c] = saddle_points[saddle_types].max()
                except ValueError:
                    # there is no unstable equilibrium
                    pass
        return unstable_equilibria, stable_equilibria


if __name__ == "__main__":
    # InfiniteNPlayerHDGDynamics.plot_hdg_gradient()
    # InfiniteNPlayerHDGDynamics.plot_hdg_equilibria()
    # hdgt = InfiniteNPlayerHDGTDynamics(N=30, c_h=0.2, R=1, c_d=0.2, T=0.4)
    InfiniteNPlayerHDGTDynamics.plot_c_h_equilibria(T=0.4)
