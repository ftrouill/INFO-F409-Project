from dataclasses import dataclass
from typing import Tuple
import numpy as np
from igraph import Graph
from numpy.lib.function_base import gradient

from game import HDG

class Lattice2d:
    """
    Class use to create and manipulate 2d lattices of individuals of 2 types (here in our case dove and hawk).

    Parameters
    ----------
    nb_element : int, number of total element in the graph
    k : int , degree = number of neighbor per vertex
    """

    def __init__(self, width: int, height: int, radius: int) -> None:
        self.w = width
        self.h = height
        self.graph = Graph.Lattice(dim=[height,width],circular=True,nei= radius)


    def fill_graph(self, population: list) -> None:
        """
        Fill the 2d graph by the sample population sent as parameter.

        Parameters
        ----------
        population : list
            Population is a list of D/H that will be placed in the graph vertices dictionnary under the key value.
        """
        if( len(population) != self.w*self.h):
            raise NameError("The given population does not match the size of the graph")
        self.graph.vs["value"] = population


    def find_k_nearest(self, id: int, k: int) -> list:
        if( (id > self.w*self.h-1) or id<0):
            raise NameError("Id is invalid")
        neis = self.graph.neighbors(id)

        if( len(neis)>k):
            neis = neis[0:k]
        return neis

    def get_neighborhood_conformation(self, id: int) -> tuple[int,int]:
        """
        Return a tuple of int containing the number of Dove and then the number of Hawk in the neighborhood

        Parameters
        ----------
        id : list, id of the vertice at for which we want to know the neighborhood.
        """

        neis = self.graph.neighbors(id)
        vertices = self.graph.vs
        nb_dove = 0
        nb_hawk = 0

        for elem in neis:
            if( vertices[elem].attributes()["value"] == "H"):
                nb_hawk += 1
            else:
                nb_dove += 1
        
        return (nb_dove, nb_hawk)


    def get_number_neigbors(self) -> int:
        return len(self.graph.neighbors(0))
    
    def get_vertices(self):
        return self.graph.vs

    
    def print_vertices(self) -> None:
        for v in self.graph.vs:
            print(v)

class InfiniteNPlayerHDGNetworkDynamic:

    def __init__(self, c_h: float, R: float, width: int, height: int, radius: int, population: list) -> None:
        # generate the network that contains all the information/method concerning the graph
        self.network = Lattice2d(width, height, radius)
        self.network.fill_graph(population)

        # Create a game with N = numbers of neigbhorhood of each vertices
        self.game = HDG(self.network.get_number_neigbors(), c_h, R)

    def update(self):
        vertices = self.network.get_vertices()
        if()
        for i in range( len(vertices):
            print(i)
            nb_dove = self.network.get_neighborhood_conformation(i)[0]
            gain = self.game.expected_payoffs(nb_dove)
            print(gain)

if __name__ == "__main__":
    D = "D"
    H = "H"
    N = [H for i in range(24)]
    N.append(D)

    w = 5
    h = 5
    radius = 1
    R = 1.0
    c_h = 0.2

    obj = InfiniteNPlayerHDGNetworkDynamic(c_h=c_h, R=R, width=w, height=h, population=N, radius=radius)
    obj.update()

    
    