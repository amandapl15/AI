# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    
   

#    print "Start's successors:", problem.getSuccessors((2,1)) #Objectivo  (1 1)


   
    first = (problem.getStartState(), None, 1)#Estado inicial (5,5)
    s = util.Stack()                        #Creamos una pila s (util.Stack contine funciones de pila (LIFO))
    s.push(first)                           #Agregamos origen a la pila s
    visit = []                              #Nodos visitados
    path = []                               #Ruta a devolver
    adyac = {}                              #Nodos adyacentes
    
    while not s.isEmpty():        
        v = s.pop()                         #sacamos un elemento de la pila s llamado v
        
        if not v[0] in visit:                    
            if problem.isGoalState(v[0]):   #Si es el objectivo delvolvemos la ruta
                             
                while v[0] != first[0]:                
                    path.append(v[1])       #Introducimos ruta de direcciones  en p               
                    v = adyac[v]            #El nodo v ahora sera el padre 
               
                path.reverse()                             
                return path                 #Devolvemos ruta al reves
            
            visit.append(v[0])
            successors = problem.getSuccessors(v[0])
            for suc in successors:          #recorremos los nodos sucesores
                if suc[0] not in visit:              
                    s.push(suc)             #Anadimos nodos no visitados a la pila                   
                    adyac[suc] = v          #Anadimos el nodo a sus hijos
                   
    return []

   
   
    
    
    
    
#    util.raiseNotDefined()
    
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #BFS es el mismo concepto que el DFS solo que cambia la Pila por la Cola, asi hara la busqueda por amplitud.
    
    first = (problem.getStartState(), None, 1)#Estado inicial 
    s = util.Queue()                        #Creamos una cola s (util.Queue contine funciones de Cola (FIFO))
    s.push(first)                           #Agregamos origen a la cola s
    visit = []                              #Nodos visitados
    path = []                               #Ruta a devolver
    adyac = {}                              #Nodos adyacentes
    
    while not s.isEmpty():        
        v = s.pop()                         #sacamos un elemento de la cola s llamado v
        
        if not v[0] in visit:                    
            if problem.isGoalState(v[0]):   #Si es el objectivo delvolvemos la ruta
                             
                while v[0] != first[0]:                
                    path.append(v[1])       #Introducimos ruta de direcciones  en p               
                    v = adyac[v]            #El nodo v ahora sera el padre 
               
                path.reverse()                              
                return path                 #Devolvemos ruta al reves
            
            visit.append(v[0])
            successors = problem.getSuccessors(v[0])
            for suc in successors:          #recorremos los nodos sucesores
                if suc[0] not in visit:              
                    s.push(suc)             #Anadimos nodos no visitados a la cola                   
                    adyac[suc] = v          #Anadimos el nodo a sus hijos
                   
    return []

  
    
    
# util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
     #El objectivo es encontrar una ruta al nodo de destino que tenga el coste acumulativo mas bajo
    
    
    first = ((problem.getStartState(), None, 0), 0)  #Estado inicial, costo inicial 0
   
    s = util.PriorityQueue()                        #Creamos una cola de prioridades s
    s.push(first,0)                                 #Agregamos origen a la cola s y prioridad
    
    visit = []                                      #Nodos visitados
    path = []                                       #Ruta a devolver
    adyac = {} 
    
    
    while not s.isEmpty():  

        v, p = s.pop()                         #sacamos un elemento de la cola y su prioridad p

        if not v[0] in visit:                    
            if problem.isGoalState(v[0]):   #Si es el objectivo delvolvemos la ruta
                             
                while v[0] != first[0][0]:                
                    path.append(v[1])       #Introducimos ruta de direcciones  en p               
                    v = adyac[v]            #El nodo v ahora sera el padre 
               
                path.reverse()                              
                return path                 #Devolvemos ruta al reves
            
            visit.append(v[0])
            successors = problem.getSuccessors(v[0])
            for suc in successors:          #recorremos los nodos sucesores
                if suc[0] not in visit: 
                    
                    cost= suc[2] + p        #Coste sucesor + coste nodo padre
                    s.push((suc,cost),cost) #Anadimos nodos no visitados a la cola y su coste acumulado                  
                    adyac[suc] = v          #Anadimos el nodo a sus hijos
                   
    return []



#  util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    #El algoritmo tiene en cuenta el coste del camino recorrido y el coste de la heuristica (estimacion hasta el objectivo)
    
    first = ((problem.getStartState(), None, 0), 0)  #Estado inicial, costo inicial 0
   
    s = util.PriorityQueue()                        #Creamos una cola de prioridades s
    s.push(first,0)                                 #Agregamos origen a la cola s y prioridad
    
    visit = []                                      #Nodos visitados
    path = []                                       #Ruta a devolver
    adyac = {} 
     
    while not s.isEmpty():  

        v, p = s.pop()                         #sacamos un elemento de la cola y su prioridad p

        if not v[0] in visit:                    
            if problem.isGoalState(v[0]):   #Si es el objectivo delvolvemos la ruta
                             
                while v[0] != first[0][0]:                
                    path.append(v[1])       #Introducimos ruta de direcciones  en p               
                    v = adyac[v]            #El nodo v ahora sera el padre 
               
                path.reverse()                              
                return path                 #Devolvemos ruta al reves
            
            visit.append(v[0])
            successors = problem.getSuccessors(v[0])
            for suc in successors:          #recorremos los nodos sucesores
                if suc[0] not in visit:                  
                    cost= suc[2] + p        #Coste sucesor + coste nodo padre
                    s.push((suc,cost),cost + heuristic(suc[0],problem)) #Anadimos nodos no visitados a la cola y su coste acumulado + su valor estimado heristico                
                    adyac[suc] = v          #Anadimos el nodo a sus hijos
                   
    return []

    

#    util.raiseNotDefined()
    
    
    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
