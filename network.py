from dataclasses import dataclass
from typing import Tuple
import numpy as np
from igraph import Graph
from numpy.lib.function_base import append, gradient

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
        Return a tuple of int containing the number of Hawk and  the number of Dove in the neighborhood

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
        
        return (nb_hawk, nb_dove)


    def get_number_neighbors(self) -> int:
        return len(self.graph.neighbors(0))

    def get_neighbors(self, index: int) -> list:
        return self.graph.neighbors(index)
    
    def get_vertices(self):
        return self.graph.vs

    
    def print_vertices(self) -> None:
        for v in self.graph.vs:
            print(v)

class InfiniteNPlayerHDGNetworkDynamic:

    CONST_HAWK = 0
    CONST_DOVE = 1
    CONST_W = 5

    def __init__(self, c_h: float, R: float, width: int, height: int, radius: int, population: list) -> None:
        # generate the network that contains all the information/method concerning the graph
        self.network = Lattice2d(width, height, radius)
        self.network.fill_graph(population)

        # Create a game with N = numbers of neigbhorhood of each vertices
        self.game = HDG(self.network.get_number_neighbors()+1, c_h, R)

    def update(self):
        gain_i_list = []
        gain_j_list = []
        change_list = []
        vertices = self.network.get_vertices()
        for i in range(len(vertices)):
            # For Hawk the strategy is zero and for Dove it is 1
            strategy = self.CONST_HAWK
            value_string = vertices[i].attributes()["value"]
            if( value_string == "D"):
                strategy = self.CONST_DOVE

            nb_dove = self.network.get_neighborhood_conformation(i)[1] + strategy
            gains = self.game.expected_payoffs(nb_dove)
            gain_i_list.append(gains[strategy])

            neis = self.network.get_neighbors(i)
            index_selected = np.random.randint(0,len(neis))
            # If the selected neighbor is of the same type we put -1 if not we put Gj 
            if(vertices[neis[index_selected]].attributes()["value"] == value_string):
                gain_j_list.append(-1) 
                change_list.append(0)
            else:
                gain_j_list.append(gains[not strategy])
                # We need to calculate the probability of changing the strategy of the vertice i p = 1/(1+exp(-w*(G_i-G_j)))
                proba = 1/(1+ np.exp(-self.CONST_W*( gains[not strategy] - gains[strategy])))
                print(proba)
                if( np.random.random() <= proba):
                    change_list.append(1)
                else:
                    change_list.append(0)

        for id,elem in enumerate(change_list):
            if(elem == 1):
                value_string = vertices[id].attributes()["value"]
                if( value_string == "D"):
                    vertices[id].attributes()["value"] == "H"
                else:
                    vertices[id].attributes()["value"] == "D"
        

if __name__ == "__main__":
    D = "D"
    H = "H"
    N = [D for i in range(24)]
    N.append(H)

    w = 5
    h = 5
    radius = 1
    R = 1.0
    c_h = 0.2

    obj = InfiniteNPlayerHDGNetworkDynamic(c_h=c_h, R=R, width=w, height=h, population=N, radius=radius)
    obj.update()

    
    