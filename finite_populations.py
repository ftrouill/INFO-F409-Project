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

    def gradient_selection(self, k: int, c_h: float, mu: float = 0.0) -> float:
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

        T_plus = (1 - mu) * pop_factor / (1 + np.exp(-weighted_diff)) + mu * ((self.Z - k)/self.Z)
        T_minus = (1 - mu) * pop_factor / (1 + np.exp(weighted_diff)) + mu * (k/self.Z)

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

        gradient = np.array([self.gradient_selection(k, c_h, mu=0) for k in k_values])

        return k_values, gradient

    def plot_gradient_selection(self, c_h: float, marker: str, edge_color: str, face_color: str, label: str):
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
        plt.plot(k_values / self.Z, gradient, marker=marker, markerfacecolor=face_color, markeredgecolor=edge_color, label=label, linestyle="None")

    @staticmethod
    def plot_hdg_gradient(N=5) -> None:
        """Plots replicator gradient for different cost values.

        Parameters
        ----------
        N : int, optional
            Sample size, by default 5
        """
        costs = [0.1, 0.5]
        Z = [10, 20, 100]
        markers = ["o", "D", "s"]
        edge_colors = ['blue', 'green', 'red']
        face_colors = {
            (10, 0.1): 'blue',
            (10, 0.5): 'None',
            (20, 0.1): 'green',
            (20, 0.5): 'None',
            (100, 0.1): 'red',
            (100, 0.5): 'None'
        }
        for i, z in enumerate(Z):
            for j, c in enumerate(costs):
                rep_dyn = FiniteNPlayerHDGDynamics(Z=z, N=N, w=1)
                rep_dyn.plot_gradient_selection(c, markers[i], edge_colors[i], face_colors[(z, c)], f'$Z = {z}, c_H = {c}$')
        plt.plot([0,1], [0,0], color='black')
        plt.xlabel(f'$k/Z$')
        plt.ylabel(f'G(k)')
        plt.xlim(0, 1)
        plt.ylim(-0.06, 0.02)
        plt.legend()
        plt.title("Gradient of selection of the N-person HDG in finite populations")
        plt.show()

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

        equilibrium = np.where(gradient[:-1] * gradient[1:] <= 0)[0]
        return equilibrium[0] if len(equilibrium) != 0 else np.nan

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

    def gradient_selection(self, k: int, c_h: float ,mu: float = 0.0) -> float:
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

        T_plus = (1 - mu) * pop_factor / (1 + np.exp(-weighted_diff)) + mu * ((self.Z - k)/self.Z)
        T_minus = (1 - mu) * pop_factor / (1 + np.exp(weighted_diff)) + mu * (k/self.Z)

        return T_plus - T_minus

    def compute_hdgt_equilibria_cost_c_h(self, resolution=100) -> Tuple[Dict, Dict]:
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
    
    def compute_hdgt_equilibria_cost_c_d(self, resolution=100) -> Tuple[Dict, Dict]:
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
            equilibria_idx = np.where((np.roll(G, -1) * G < 0) | (np.abs(G) <= epsilon))[0]
            saddle_points = equilibria_idx / (self.Z)
            saddle_types = find_saddle_type_and_gradient_direction(
                G, equilibria_idx
            )[0]
            
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
            """if (0.0 in st_eq and 1.0 in st_eq):
                plt.plot(
                st_eq.keys(),
                st_eq.values(),
                linestyle='--',
                color=color,
            )
            if (0.0 in un_eq and 1.0 in un_eq):
                plt.plot(
                un_eq.keys(),
                un_eq.values(),
                linestyle='--',
                color=color,
            )"""
            plt.plot(
                st_eq.keys(),
                st_eq.values(),
                linestyle='None',
                marker='o',
                label=f"$c_D={c_d}$ - stable",
                color=color,
            )
            plt.plot(
                un_eq.keys(),
                un_eq.values(),
                linestyle='None',
                marker='o',
                markerfacecolor='None',
                label=f"$c_D={c_d}$ - unstable",
                color=color,
            )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel(f'$c_H$')
        plt.ylabel(f'$k^*/Z$')
        plt.legend()
        #plt.savefig(f"figures/raw/fig5_mu/c_h_0_{str(T)[-1]}.png")
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
            """if (0.0 in st_eq and 1.0 in st_eq):
                plt.plot(
                st_eq.keys(),
                st_eq.values(),
                linestyle='--',
                color=color,
            )
            if (0.0 in un_eq and 1.0 in un_eq):
                plt.plot(
                un_eq.keys(),
                un_eq.values(),
                linestyle='--',
                color=color,
            )"""
            plt.plot(
                st_eq.keys(),
                st_eq.values(),
                linestyle='None',
                marker='o',
                label=f"$c_H={c_h}$ - stable",
                color=color,
            )
            plt.plot(
                un_eq.keys(),
                un_eq.values(),
                linestyle='None',
                marker='o',
                markerfacecolor='None',
                label=f"$c_H={c_h}$ - unstable",
                color=color,
            )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel(f'$c_D$')
        plt.ylabel(f'$k^*/Z$')
        plt.legend()
        #plt.savefig(f"figures/raw/fig5_mu/c_D_0_{str(T)[-1]}.png")
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
        plt.plot(ch_values, equilibria/Z, label=f'N = {N}', marker='D', markerfacecolor='None')

    plt.xlabel("$C_H$")
    plt.ylabel("$k^*/Z$")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Equilibria of the N-person HDG in finite populations")
    plt.legend()
    #plt.savefig("plots/equilibria.png")
    plt.show()


if __name__ == "__main__":
    #finite = FiniteNPlayerHDGDynamics(Z=100, )
    #FiniteNPlayerHDGDynamics.plot_hdg_gradient()
    #fig, gradient = finite.plot_gradient_selection(0.5)
    #plt.show()
    #gradient[40] = 0.0
    #print(gradient[39:44])
    #print(finite.find_equilibrium(gradient))
    #plot_equilibria(100, 1)
    for T in [0.2, 0.4, 0.6, 0.8]:
        FiniteNPlayerHDGTDynamics.plot_c_d_equilibria(T=T)



