import matplotlib
import numpy as np
from igraph import Graph
from numpy.core.fromnumeric import mean, size
import matplotlib.pyplot as plt


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
        if( (id > self.w*self.h-1) or id<0):
            raise NameError("Id is invalid")

        neis = self.graph.neighbors(id)
        vertices = self.graph.vs
        nb_dove = 0
        nb_hawk = 0

        for elem in neis:
            nb_dove += vertices[elem].attributes()["value"]
        nb_hawk = len(neis)-nb_dove
        
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
    """
    Class that will generate a network as a 2dLattice and include the methods used to update the graph following a game 
    payoff matrix.

    Parameters
    ----------
    c_h : float, Cost for the hawks
    R : float, Reward
    width : int, dim of the 2dlattice
    height : int, dim of the 2dlattice
    radius : int, radius use to generate edges ine the lattice (cf lattice)
    population : list, list of values to put in the graph (each vertice will contain a value)
    """

    CONST_HAWK = 0
    CONST_DOVE = 1
    CONST_W = 5

    def __init__(self, c_h: float, R: float, width: int, height: int, radius: int, population: list) -> None:
        # generate the network that contains all the information/method concerning the graph
        self.network = Lattice2d(width, height, radius)
        self.network.fill_graph(population)

        # Create a game with N = numbers of neigbhorhood of each vertices  + the one being tested
        self.game = HDG(self.network.get_number_neighbors()+1, c_h, R)

    def update(self):
        """
        Update the graph vertices values by following the process:
            - Calculate a G_i (gain) for the vertice i by playing against all the neighbor link to this vertex
            - Select a vertex j in the neighborhood
            - If the strategy of j !=i then we calculate G_j as if i was of type j
            - Calculate the probablity of replacing i by j by following a fermi function
            - Update the graph vertices with the new list of values
        """

        gain_i_list = []
        gain_j_list = []
        change_list = []
        vertices = self.network.get_vertices()
        for i in range(len(vertices)):
            # For Hawk the strategy is zero and for Dove it is 1
            strategy = vertices[i].attributes()["value"]

            nb_dove = self.network.get_neighborhood_conformation(i)[1] + strategy
            gains = self.game.expected_payoffs(nb_dove)
            gain_i_list.append(gains[strategy])

            neis = self.network.get_neighbors(i)
            index_selected = np.random.randint(0,len(neis))
            # If the selected neighbor is of the same type we put -1 if not we put Gj 
            if(vertices[neis[index_selected]].attributes()["value"] == strategy):
                gain_j_list.append(-1) 
                change_list.append(0)
            else:
                gain_j_list.append(gains[not strategy])
                # We need to calculate the probability of changing the strategy of the vertice i p = 1/(1+exp(-w*(G_i-G_j)))
                proba = 1/(1+ np.exp(-self.CONST_W*( gains[not strategy] - gains[strategy])))
                if( np.random.random() <= proba):
                    change_list.append(1)
                else:
                    change_list.append(0)

        new_pop = []
        for id,elem in enumerate(change_list):
            new_strat = vertices[id].attributes()["value"]
            if(elem == 1):
                new_strat = int(not (new_strat))
            new_pop.append(new_strat)

        self.network.fill_graph(new_pop)

    def calculate_dove_ratio(self):
        value_list = self.network.graph.vs()["value"]
        return sum(value_list)/len(value_list)

    @staticmethod
    def generate_population(size: int, ratio_dove: float) -> list:
        if( ratio_dove > 1 or ratio_dove < 0):
            raise NameError("Invalid ratio of doves")
    
        binomial_samples = np.random.binomial(1,ratio_dove,size)

        return binomial_samples



if __name__ == "__main__":

    if False: 
        w = 50
        h = 50
        x = 0.5
        radius_list = [i for i in range(1,5)]
        R = 1.0
        c_h_list = [i/100 for i in range(101)]
        results = []
        

        nb_saved = 5
        steps = 40

        results = []
        N_list = []
        for radius in radius_list:
            result = []
            # We iterate over all values of c_h and simulate for each the comportement of a population randomly sample with a
            # fixed ratio of dove/hawk and save the ratio of dove at the end of each loop of a c_h
            for elem in c_h_list:
                pop = InfiniteNPlayerHDGNetworkDynamic.generate_population(w*h, x)
                obj = InfiniteNPlayerHDGNetworkDynamic(c_h=elem, R=R, width=w, height=h, population=pop, radius=radius)

                list_values = []
                for i in range(steps):
                    if(i >= steps-nb_saved):
                        list_values.append(obj.calculate_dove_ratio())
                    obj.update()

                result.append(np.around(mean(list_values),decimals=4))
                print("End of step : " + str(elem) )
            
            results.append(result)
            N_list.append(obj.network.get_number_neighbors())
        print("Neighbor numbers list")
        print(N_list)

        plt.figure(figsize=(10,10))
        plt.axis([0,1,0,1])
        plt.xlabel("c_h")
        plt.ylabel("x*")
        for elem in results:
            plt.plot(c_h_list, elem)
            plt.scatter(c_h_list, elem)
        plt.show()

    
    