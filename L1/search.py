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

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # Keep track of nodes already expanded
    expanded_nodes = []
    # In DFS, frontier is a Stack (nodes not expanded yet)
    frontier = util.Stack()
    # Add first state to frontier, add also their action and cost of this action
    first_state = (problem.getStartState(), [], 1)
    # Add in frontier the first state and the parent (we do not have parent yet,
    # so is None)
    frontier.push((first_state, None))
    # Keep track of the path
    path = []
    while True:
        # If there is no frontier, we did not find a solution in expanded nodes
        # so we return no solution path
        if frontier.isEmpty():
            return []
        # Remove and obtain node n
        n, parent = frontier.pop()
        if n[0] not in expanded_nodes:
            # Add n to expanded nodes
            expanded_nodes.append(n[0])
            # If n is a goal state return the solution path by going backwards
            # from this node
            if problem.isGoalState(n[0]):
                # Initialize curr_state, here we store the node that we want to
                # get the action
                curr_state = n
                # When the curr_state is the first one we are done
                while curr_state != first_state:
                    path.insert(0, curr_state[1])
                    # Now when we add the action we need to do curr_state = the
                    # parent of himself, so there we can take again the action
                    curr_state = parent[0]
                    # The parent of the curr_state
                    parent = parent[1]
                return path
            # In these lines expand the node n
            for child in problem.getSuccessors(n[0]):
                if child[0] not in expanded_nodes:
                    # If a node its not in the frontier, add the node and the
                    # parent node and his respective parent, so we can go backwards
                    # when we find a solution
                    frontier.push((child, (n, parent)))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Keep track nodes already expanded
    expanded_nodes = []
    # In BFS, frontier is a Queue
    frontier = util.Queue()
    # firstState inicialization and push to the queue
    first_state = (problem.getStartState(), [], 1)
    frontier.push((first_state, None))
    # We have to keep track of the path
    path = []
    while True:
        # Comprobation for empty frontier
        if frontier.isEmpty():
            return []
        # We take the oldest nodes with a pop and add the expanded nodes
        # at the end of the queue
        # Remove and obtain node n
        n, parent = frontier.pop()
        if n[0] not in expanded_nodes:
            # Add n to expanded nodes
            expanded_nodes.append(n[0])
            if problem.isGoalState(n[0]):
                # Initialize curr_state
                curr_state = n
                while curr_state != first_state:
                    path.insert(0, curr_state[1])
                    curr_state = parent[0]
                    parent = parent[1]
                return path
            # In these lines expand the node n
            for child in problem.getSuccessors(n[0]):
                if child[0] not in expanded_nodes:
                    # If a node its not in the frontier, add the node and the
                    # parent node and his respective parent, so we can go backwards
                    # when we find a solution
                    frontier.push((child, (n, parent)))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Keep track nodes already expanded
    expanded_nodes = []
    # In UC, frontier is a PriorityQueue
    frontier = util.PriorityQueue()
    # Add first state to frontier, action and cost action
    first_state = (problem.getStartState(), [], 0)
    frontier.push((first_state, None), 0)
    # We have to keep track of the path
    path = []
    while True:
        # Comprobation for empty frontier
        if frontier.isEmpty():
            return []
        # Remove and obtain node n
        n, parent = frontier.pop()
        if n[0] not in expanded_nodes:
            # Add n to expanded nodes
            expanded_nodes.append(n[0])
            if problem.isGoalState(n[0]):
                # Initialize curr_state
                curr_state = n
                while curr_state != first_state:
                    path.insert(0, curr_state[1])
                    curr_state = parent[0]
                    parent = parent[1]
                return path
            # In these lines expand the node n
            for child in problem.getSuccessors(n[0]):
                if child[0] not in expanded_nodes:
                    # If a node its not in the frontier and
                    # in expanded nodes, we need to push him in the stack
                    # In uniform cost we need to keep track the f function,
                    # which is the cost of the path without the heuristic (g(n))
                    # so the priority of that element is that value, and the cost of the action too,
                    # with that we can compute new f on the successors
                    frontier.push(((child[0], child[1], child[2] + n[2]), (n, parent)), child[2] + n[2])

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Keep track nodes already expanded
    expanded_nodes = []
    # In A*, like UC, frontier is a PriorityQueue
    frontier = util.PriorityQueue()
    # Add first state to frontier, action and cost action
    first_state = (problem.getStartState(), [], 0)
    frontier.push((first_state, None), heuristic(first_state[0], problem))
    # We have to keep track of the path
    path = []
    while True:
        # Comprobation for empty frontier
        if frontier.isEmpty():
            return []
        # Remove and obtain node n
        n, parent = frontier.pop()
        if n[0] not in expanded_nodes:
            expanded_nodes.append(n[0])
            if problem.isGoalState(n[0]):
                curr_state = n
                while curr_state != first_state:
                    path.insert(0, curr_state[1])
                    curr_state = parent[0]
                    parent = parent[1]
                return path
            for child in problem.getSuccessors(n[0]):
                if child[0] not in expanded_nodes:
                    # In a star we need to keep track the f function,
                    # which is the cost of the path plus the heuristic (g(n)+h(n))
                    # so the priority of that element is that value, and the cost of the action is
                    # the value of g(n), with that we can compute the values of f without adding
                    # heuristics of the previous nodes
                    frontier.push(((child[0], child[1], child[2] + n[2]), (n, parent)), child[2] + n[2] + heuristic(child[0], problem))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
