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
                label=f"$c_d={c_d}$",
                color=color,
            )
            plt.plot(
                un_eq.keys(),
                un_eq.values(),
                "--",
                label=f"$c_d={c_d}$",
                color=color,
            )
        plt.legend()
        plt.show()

    @staticmethod
    def plot_c_d_equilibria(N=5, T=0.2):
        c_h_values = [0.2, 0.5, 0.8]
        colors = ["red", "blue", "black"]
        for c_h, color in zip(c_h_values, colors):
            print(c_h)
            rep_dyn = InfiniteNPlayerHDGTDynamics(N, c_h, nb_states=10000, T=T)
            un_eq, st_eq = rep_dyn.compute_hdgt_equilibria_cost_c_d()
            plt.plot(
                st_eq.keys(),
                st_eq.values(),
                label=f"$c_H={c_h}$",
                color=color,
            )
            plt.plot(
                un_eq.keys(),
                un_eq.values(),
                "--",
                label=f"$c_H={c_h}$",
                color=color,
            )
        plt.legend()
        plt.show()

    def compute_hdgt_equilibria_cost_c_h(self) -> Tuple[Dict, Dict]:
        """Computes stable and unstable equilibria for different c_h values.

        Returns
        -------
        Tuple[Dict, Dict]
            Unstable and stable equilibria as dict(cost, equilibria)
        """
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

    def compute_hdgt_equilibria_cost_c_d(self) -> Tuple[Dict, Dict]:
        """Computes stable and unstable equilibria for different c_d values.

        Returns
        -------
        Tuple[Dict, Dict]
            Unstable and stable equilibria as dict(cost, equilibria)
        """
        # array of different costs
        costs = np.linspace(0, 1, num=self.nb_costs, dtype=np.float64)
        # array of states
        dove_strategy = np.linspace(0, 1, num=self.nb_states, dtype=np.float64)
        # find optimal state for each cost
        unstable_equilibria = dict()
        stable_equilibria = dict()
        for i, c in enumerate(costs):
            # update cost
            self.c_d = c
            # compute gradient for each state
            G = np.array([self.compute_gradient_for_state(i) for i in dove_strategy])
            epsilon = 1e-6
            saddle_points_idx = np.where(
                (np.roll(G, -1) * G < 0) | (np.abs(G) <= epsilon)
            )[0]
            saddle_points = saddle_points_idx / (self.nb_states - 1)
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

    def compute_stable_equilibria(self, epsilon=1e-6) -> list:
        """
        Compute the stable equilibria for the current game

        :param epsilon: tolerance for equilibrium points
        :return: array of stable equilibria as fraction of doves
        """
        # array of states
        dove_strategy = np.linspace(0, 1, num=self.nb_states, dtype=np.float64)
        # find optimal state for each cost
        equilibria = []
        G = np.array([self.compute_gradient_for_state(i) for i in dove_strategy])
        equilibria_idx = np.where((np.roll(G, -1) * G < 0) | (np.abs(G) <= epsilon))[0]
        equilibria = equilibria_idx / (self.nb_states - 1)
        # Check for x = 0 and x = 1
        if G[1] > 0:
            # We know G[0] = 0, so we check if it goes directly down with G[1] -> Unstable point
            # Remove x = 0
            equilibria = np.delete(equilibria, 0)
        if G[-2] < 0:
            # We know G[-1] = 0, so we check if it goes directly up with G[-2] -> Unstable point
            # Remove x = 1
            equilibria = np.delete(equilibria, -1)
        return equilibria

    @staticmethod
    def get_phase(equilibria, epsilon=1e-6) -> Tuple[bool, bool, bool]:
        """
        Give the phase based on the equilibria

        :param equilibria: array of equilibria as fraction of doves
        :param epsilon: tolerance for bi-stable states
        :return: tuple of booleans to describe the phase (Doves, Hawks, Mixed)
        """
        has_full_dove = False
        has_full_hawk = False
        has_mixed = False
        for equilibrium in equilibria:
            if equilibrium > 1 - epsilon:
                has_full_dove = True
            elif equilibrium < epsilon:
                has_full_hawk = True
            else:
                has_mixed = True
        return has_full_dove, has_full_hawk, has_mixed

    @staticmethod
    def plot_phase_diagram(N, T, resolution_cost=100) -> None:
        """
        Plot the phase diagram for the given sample size N,
        the threshold T and the resolution of the graph

        :param N: sample size
        :param T: threshold for doves
        :param resolution_cost: will generate a grid of
            resolution_cost x resolution_cost points
        """
        c_d_values = np.linspace(0, 1, resolution_cost)
        c_h_values = np.linspace(0, 1, resolution_cost)
        phase_dictionary = {
            # (Doves, Hawks, Mixed)
            (True, True, True): ("hawks + doves + mixed", "blue"),
            (True, True, False): ("hawks + doves", "orange"),
            (True, False, True): ("doves + mixed", "green"),
            (True, False, False): ("doves", "red"),
            (False, True, True): ("hawks + mixed", "purple"),
            (False, True, False): ("hawks", "brown"),
            (False, False, True): ("mixed", "pink"),
            (False, False, False): ("no equilibrium", "grey"),
        }
        already_labeled = {
            (True, True, True): False,
            (True, True, False): False,
            (True, False, True): False,
            (True, False, False): False,
            (False, True, True): False,
            (False, True, False): False,
            (False, False, True): False,
            (False, False, False): False,
        }
        for c_d in c_d_values:
            for c_h in c_h_values:
                rep_dyn = InfiniteNPlayerHDGTDynamics(
                    N, c_h, nb_states=100, T=T, c_d=c_d
                )
                equilibria = rep_dyn.compute_stable_equilibria()
                phase = rep_dyn.get_phase(equilibria)
                color = phase_dictionary[phase][1]
                if not already_labeled[phase]:
                    already_labeled[phase] = True
                    plt.plot(
                        [c_h],
                        [c_d],
                        label=phase_dictionary[phase][0],
                        marker=".",
                        markersize=10,
                        color=color,
                    )
                else:
                    plt.plot(
                        [c_h],
                        [c_d],
                        marker=".",
                        markersize=500 / resolution_cost,
                        color=color,
                    )
        plt.title(f"T = {T}")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # InfiniteNPlayerHDGDynamics.plot_hdg_gradient()
    # InfiniteNPlayerHDGDynamics.plot_hdg_equilibria()
    # hdgt = InfiniteNPlayerHDGTDynamics(N=30, c_h=0.2, R=1, c_d=0.2, T=0.4)
    InfiniteNPlayerHDGTDynamics.plot_c_h_equilibria(T=0.4)
    # InfiniteNPlayerHDGTDynamics.plot_c_d_equilibria(T=0.4)
    # InfiniteNPlayerHDGTDynamics.plot_c_h_equilibria(T=0.6)
    # InfiniteNPlayerHDGTDynamics.plot_phase_diagram(N = 5, T = 0.4, resolution_cost = 50)
