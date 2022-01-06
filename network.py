import igraph
import matplotlib
import numpy as np
from igraph import Graph
from numpy.core.fromnumeric import mean, size
import matplotlib.pyplot as plt
from numpy.random.bit_generator import SeedlessSeedSequence


from game import HDG


class Lattice2d:
    """
    Class use to create and manipulate 2d lattices of individuals of 2 types (here in our case dove and hawk).

    Parameters
    ----------
    width : int, dim of the 2dlattice
    height : int, dim of the 2dlattice
    radius : int, radius use to generate edges ine the lattice (cf lattice)
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

    def change_edges(self, p: float) -> None:
        """
        Update the graph by selecting each edge and changing it by an edge between two random vertices with a proba p.
        We will not choose edges that leaves isolated points.

        Parameters
        ----------
        p : float, proba of changing an edge
        """
        edges = [] 
        for elem in self.graph.es:
            edges.append(elem.tuple)

        id_limit = len(self.graph.vs)
        neighbors_numbers = [len(self.get_neighbors(i)) for i in range(id_limit)]
        replaced_edges = []
        new_edges = []
        for id, e in enumerate(edges):
            if(np.random.random() <= p):
                replaced_edges.append(e)
                end = False
                id_v1 = 0
                id_v2 = 0
                edge = (0,0)
                while( not end):
                    id_v1 = np.random.randint(0,id_limit)
                    id_v2 = np.random.randint(0,id_limit)
                    edge = (id_v1, id_v2)
                    if( (id_v1 != id_v2) and (not edge in edges) and (not edge in new_edges)):
                        end = True
                        new_edges.append(edge)

        for new_edge,edge in zip(new_edges,replaced_edges):
            # We make sure that no point is left alone without edge
            if ( (neighbors_numbers[edge[0]] > 1) and (neighbors_numbers[edge[1]] > 1) ):
                self.graph.delete_edges([edge])
                self.graph.add_edges([new_edge])
                # We update the number of neighbors
                neighbors_numbers[edge[0]] -= 1
                neighbors_numbers[edge[1]] -= 1
                neighbors_numbers[new_edge[0]] += 1
                neighbors_numbers[new_edge[1]] += 1
                


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
    mode : int, 0 for lattice only and 1 for small world
    """

    CONST_HAWK = 0
    CONST_DOVE = 1
    CONST_W = 5
    CONST_MODE_LATTICE = 0
    CONST_MODE_SW = 1
    CONST_PROBA_EDGE_SW = 0.05 # Probrability of an edge changing in the small world sim

    def __init__(self, c_h: float, R: float, width: int, height: int, radius: int, population: list, mode: int) -> None:
        # generate the network that contains all the information/method concerning the graph
        self.network = Lattice2d(width, height, radius)
        self.network.fill_graph(population)
        self.mode = mode

        # Create a game with N = numbers of neigbhorhood of each vertices  + the one being tested which is fixed for the Lattice
        # but the N will change for each loop and vertex when using smal world (cf update)
        self.game = HDG(len(self.network.get_neighbors(0))+1, c_h, R)

    def update(self) -> None:
        """
        Update the graph vertices values by following the process:
            - If small world mode update the graph by changing the edges following a fixed probability
            - Calculate a G_i (gain) for the vertice i by playing against all the neighbor link to this vertex
            - Select a vertex j in the neighborhood
            - If the strategy of j !=i then we calculate G_j as if i was of type j
            - Calculate the probablity of replacing i by j by following a fermi function
            - Update the graph vertices with the new list of values
        """
        # If small world we change the edges
        if( self.mode == self.CONST_MODE_SW):
            self.network.change_edges(self.CONST_PROBA_EDGE_SW)

        gain_i_list = []
        gain_j_list = []
        new_pop = []
        vertices = self.network.get_vertices()
        for i in range(len(vertices)):
            # For Hawk the strategy is zero and for Dove it is 1
            strategy = vertices[i].attributes()["value"]

            nb_dove = self.network.get_neighborhood_conformation(i)[1] + strategy

            # If we are in the small world configuration we need to update N of the game before calculating the payoff
            if(self.mode == self.CONST_MODE_SW):
                self.game.set_N(len(self.network.get_neighbors(i))+1)

            gains = self.game.expected_payoffs(nb_dove)
            gain_i_list.append(gains[strategy])

            neis = self.network.get_neighbors(i)
            index_selected = np.random.randint(0,len(neis))
            # If the selected neighbor is of the same type we put -1 if not we put Gj 
            if(vertices[neis[index_selected]].attributes()["value"] == strategy):
                gain_j_list.append(-1) 
                new_pop.append(strategy)
            else:
                gain_j_list.append(gains[not strategy])
                # We need to calculate the probability of changing the strategy of the vertice i p = 1/(1+exp(-w*(G_i-G_j)))
                proba = 1/(1+ np.exp(-self.CONST_W*( gains[not strategy] - gains[strategy])))
                if( np.random.random() <= proba):
                    new_pop.append(not strategy)

                else:
                    new_pop.append(strategy)

        self.network.fill_graph(new_pop)

    def calculate_dove_ratio(self) -> float:
        value_list = self.network.graph.vs()["value"]
        return sum(value_list)/len(value_list)

    @staticmethod
    def generate_population(size: int, ratio_dove: float) -> list:
        if( ratio_dove > 1 or ratio_dove < 0):
            raise NameError("Invalid ratio of doves")
    
        binomial_samples = np.random.binomial(1,ratio_dove,size)

        return binomial_samples




if __name__ == "__main__":

    if True: 
        w = 25
        h = 25
        x = 0.5
        radius_list = [i for i in range(1,5)]
        R = 1.0
        c_h_list = [i/100 for i in range(0,101)]
        results = []
        mode = InfiniteNPlayerHDGNetworkDynamic.CONST_MODE_SW
        

        nb_saved = 10
        steps = 50

        results = []
        for radius in radius_list:
            result = []
            # We iterate over all values of c_h and simulate for each the comportement of a population randomly sample with a
            # fixed ratio of dove/hawk and save the ratio of dove at the end of each loop of a c_h
            for elem in c_h_list:
                pop = InfiniteNPlayerHDGNetworkDynamic.generate_population(w*h, x)
                obj = InfiniteNPlayerHDGNetworkDynamic(c_h=elem, R=R, width=w, height=h, population=pop, radius=radius, mode=mode)

                list_values = []
                for i in range(steps):
                    if(i >= steps-nb_saved):
                        list_values.append(obj.calculate_dove_ratio())
                    obj.update()

                result.append(np.around(mean(list_values),decimals=4))
                print("End of step : " + str(elem) )
            
            results.append(result)

        N = [5, 13, 25, 41]

        plt.figure(figsize=(10,10))
        if(mode == InfiniteNPlayerHDGNetworkDynamic.CONST_MODE_LATTICE):
            plt.title("Equilibria of the HDG in 2d lattice network")
        elif( mode == InfiniteNPlayerHDGNetworkDynamic.CONST_MODE_SW):
            plt.title("Equilibria of the HDG in small world network")
        plt.axis([0,1,0,1])
        font = {'family':'serif','size':15}
        plt.xlabel("C_h",fontdict=font)
        plt.ylabel("x*",fontdict=font)
        for id,elem in enumerate(results):
            plt.plot(c_h_list, elem, label="N = "+str(N[id]))
            plt.scatter(c_h_list, elem)
        plt.legend(loc="lower right",fontsize="x-large")
        plt.show()

    
    