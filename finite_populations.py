import numpy as np
from game import HDG, HDG_T
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from egttools.utils import find_saddle_type_and_gradient_direction


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

@dataclass
class FiniteNPlayerHDGTDynamics(FiniteNPlayerHDGDynamics):
    """Generates replicator dynamics for HDGT with finite population.

    Parameters
    ----------
    c_d : float
        Cost for doves to protect the resource.
    T : float
        Doves threshold to protect the resource.
    """

    c_d: float = 0.5
    T: float = 0.5

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

        fh, fd = HDG_T(self.N, c_h, c_d=self.c_d, T=self.T).average_fitness_finite_pop(k, self.Z)

        weighted_diff = self.w * (fd - fh)
        pop_factor = (k * (self.Z - k)) / (self.Z ** 2)

        T_plus = pop_factor / (1 + np.exp(-weighted_diff))
        T_minus = pop_factor / (1 + np.exp(weighted_diff))

        return T_plus - T_minus

    def compute_hdgt_equilibria_cost_c_h(self, resolution=1000) -> Tuple[Dict, Dict]:
        """Computes stable and unstable equilibria for different c_h values.

        Returns
        -------
        Tuple[Dict, Dict]
            Unstable and stable equilibria as dict(cost, equilibria)
        """
        # array of different costs
        costs = np.linspace(0, 1, num=resolution, dtype=np.float64)
        # find optimal state for each cost
        unstable_equilibria = dict()
        stable_equilibria = dict()
        for i, c in enumerate(costs):
            # update cost
            self.c_h = c
            # compute gradient for each state
            _, G = self.compute_full_gradient(c)
            G[0] = 0
            G[-1] = 0
            epsilon = 1e-6
            saddle_points_idx = np.where(
                (np.roll(G, -1) * G < 0) | (np.abs(G) <= epsilon)
            )[0]
            saddle_points = saddle_points_idx / self.Z

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

    def compute_hdgt_equilibria_cost_c_d(self, resolution=1000) -> Tuple[Dict, Dict]:
        """Computes stable and unstable equilibria for different c_d values.

        Returns
        -------
        Tuple[Dict, Dict]
            Unstable and stable equilibria as dict(cost, equilibria)
        """
        # array of different costs
        costs = np.linspace(0, 1, num=resolution, dtype=np.float64)
        # find optimal state for each cost
        unstable_equilibria = dict()
        stable_equilibria = dict()
        for i, c in enumerate(costs):
            # update cost
            self.c_d = c
            # compute gradient for each state
            _, G = self.compute_full_gradient(self.c_h)
            G[0] = 0
            G[-1] = 0
            epsilon = 1e-6
            saddle_points_idx = np.where(
                (np.roll(G, -1) * G < 0) | (np.abs(G) <= epsilon)
            )[0]
            saddle_points = saddle_points_idx / self.Z

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
    
    @staticmethod
    def plot_c_h_equilibria(Z = 100, N=5, T=0.2):
        c_d_values = [0.2, 0.5, 0.8]
        colors = ["red", "blue", "black"]
        for c_d, color in zip(c_d_values, colors):
            print(c_d)
            rep_dyn = FiniteNPlayerHDGTDynamics(Z=Z, N=N, w=1.0, T=T, c_d=c_d)
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
    def plot_c_d_equilibria(Z = 100, N=5, T=0.2):
        c_h_values = [0.2, 0.5, 0.8]
        colors = ["red", "blue", "black"]
        for c_h, color in zip(c_h_values, colors):
            print(c_h)
            rep_dyn = FiniteNPlayerHDGTDynamics(Z=Z, N=N, w=1.0, T=T)
            rep_dyn.c_h = c_h
            un_eq, st_eq = rep_dyn.compute_hdgt_equilibria_cost_c_d()
            plt.plot(
                st_eq.keys(),
                st_eq.values(),
                label=f"$c_h={c_h}$",
                color=color,
            )
            plt.plot(
                un_eq.keys(),
                un_eq.values(),
                "--",
                label=f"$c_h={c_h}$",
                color=color,
            )
        plt.legend()
        plt.show()

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
    #finite = FiniteNPlayerHDGDynamics(100, 5, 1)
    #fig, gradient = finite.plot_gradient_selection(0.5)
    #plt.show()
    #gradient[40] = 0.0
    #print(gradient[39:44])
    #print(finite.find_equilibrium(gradient))
    #plot_equilibria(100, 1)
    FiniteNPlayerHDGTDynamics.plot_c_d_equilibria(T=0.4)



