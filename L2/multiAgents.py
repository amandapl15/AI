# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        # Init for all the food and their distance, manipulation with a list
        distFood = float("inf")
        currentFood = currentGameState.getFood()
        foodList = currentFood.asList()
        # Take ghosts positions as well
        ghostsPositions = currentGameState.getGhostPositions()
        
        # Initialize ghost, as non scared
        for ghost in range(len(newScaredTimes)):
            if newScaredTimes[ghost] == 0 and newPos == ghostsPositions[ghost]:
                return float("-inf")
                
        # Food inicialization with the minimal distances
        for food in foodList:
            distFood = min(distFood, util.manhattanDistance(food, newPos))

        # We substract the value of the distance because more far is the food less value we get, and when its near the value must be higher
        return successorGameState.getScore() - distFood


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(agent, depth, gameState):
            # 'Base' check for the init
            if gameState.isLose() or gameState.isWin() or depth == 0:
                # Return evaluation function when a terminal state is reached (lose or win) or the depth is 0
                v = self.evaluationFunction(gameState)
                return v

            if agent == 0:
                # Maximize for pacman, next player is ghost 1
                v = max(minimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in
                        gameState.getLegalActions(agent))
                return v

            else:
                # Minimize for ghosts, the next agent can be a ghost or Pacman, so if gameState.getNumAgents() =
                # nextAgent the next agent will be the pacman
                nextAgent = agent + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                    depth = depth - 1  # We decrease the depth now because a single search ply is considered to be one
                    # Pacman move and all the ghosts’ responses
                # Compute the minimal value of the successors
                v = min(minimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in
                        gameState.getLegalActions(agent))
                return v
                
        # Minimax first node tree init
        maximum = float("-inf")  # Current best value of a move
        bestAction = Directions.WEST # Current best action (by default)
        
        # Current execution for the childs with actions that pacman can do in that initial gameState,
        # in this for we do a computation like in minimax when the agent is 0, so we can check what is the best
        # action that pacman can do, its necessary because in our implementation of minimax algorithm we only return
        # the value, so we do not know what its the best action if we simply call with an agent 0
        for action in gameState.getLegalActions(0):  

            suc = gameState.generateSuccessor(0, action) # The state of the following ghost, when an action is chosen
            val = minimax(1, self.depth, suc) # Value returned of minimax function
            
            if val >= maximum: # We check what is the action that gives the best value and its returned
                maximum = val
                bestAction = action
        
        # Return of the minimax result
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def maximizer(agent, depth, game_state, alpha, beta):
            # Initialize the score, this will store the best pacman score
            score = float("-inf")
            
            # Child state loop to return the higher value of the next successor
            for newState in game_state.getLegalActions(agent): 

                suc = game_state.generateSuccessor(agent, newState)
                score = max(score, alphaBetaPrune(1, depth, suc, alpha, beta))
                # If the value is higher than beta we can prune because the next values that pacman 
                # would check for sure the ghost parent wont choose it
                if score > beta:
                    break
                    
                alpha = max(alpha, score) # Update alpha with the minimal value in this moment
            
            # Finall reurn for this score
            return score

        # To get the minimum instead of the maximized score
        def minimizer(agent, depth, game_state, alpha, beta):
            # Initialize the score
            score = float("inf")
            # Set agents
            next_agent = agent + 1
            # We decrease the depth now because a single search ply is considered to be one
            # pacman move and all the ghosts' responses
            if game_state.getNumAgents() == next_agent:
                next_agent = 0
                depth = depth + 1  # Add depth
            
            # Child state loop
            for newState in game_state.getLegalActions(agent):

                suc = game_state.generateSuccessor(agent, newState)
                score = min(score, alphaBetaPrune(next_agent, depth, suc, alpha, beta))
                # If the value is less than alpha we can prune because the next values that this ghost 
                # would check for sure the pacman parent wont choose it
                if score < alpha:
                    break
                beta = min(beta, score) # Update beta with the minimal value in this moment
            
            # Finall reurn for this score
            return score

        # Almost the same structure that minimax from above, but in this case we have values alpha and beta
        # that possibilities the capacity to stop when its not necessary to continue searching 
        # for successors and make alpha or betha cut
        def alphaBetaPrune(agent, depth, game_state, alpha, beta): 
            
            # Firts the initial check for a terminal state
            if game_state.isLose() or game_state.isWin() or depth == self.depth:
                # Return evaluation function when a terminal state is reached (lose or win) or the depth is the max
                return self.evaluationFunction(game_state)
            
            # Setting for agent and ghosts too
            if agent == 0:
                # Maximize for Pacman
                return maximizer(agent, depth, game_state, alpha, beta)
                
            else:
                # Minimize for ghosts
                return minimizer(agent, depth, game_state, alpha, beta)

        action = Directions.WEST # Current best action by default
        # Like in the minimax function we need to compute an iteration of alphabeta because we only
        # return the value with no action, and this is the current best value that we well take as
        # initial value 
        alpha = float("-inf")
        beta = float("inf")
        val = float("-inf")
        
        # Loop for the agent tree state development
        for agentState in gameState.getLegalActions(0):
            # Generates the gamestate of ghost 1
            suc = gameState.generateSuccessor(0, agentState) 
            # Obtain the value of the successor, so we can check which
            # is the action that gives the higher value and return it
            ghostVal = alphaBetaPrune(1, 0, suc, alpha, beta)

            if ghostVal > val:
                val = ghostVal
                action = agentState
            # If the value is higher than beta we can make a cut because the next values that pacman would
            # check for sure the ghost parent wont choose it (like in alphabeta)
            if val > beta: 
                return val
            # Update alpha with the higher value
            alpha = max(alpha, val) 
            
        # Final return for the action
        return action



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(agent, depth, gameState):
            
            # Initial check
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
                
            # Maximize for the pacman agent
            if agent == 0:
                return max(expectimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in
                           gameState.getLegalActions(agent))
                           
            else:
                # In this case, do not take minim and instead take the expectation
                nextAgent = agent + 1  
                # Pass to the next agent
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth = depth + 1  # Add depth
                    
                return sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in
                           gameState.getLegalActions(agent)) / float(len(gameState.getLegalActions(agent)))
        
        # Initialize the max storage value
        maximum = float("-inf")
        action = Directions.WEST # Current best action by default
        
        # Loop for the agent tree state development
        for agentState in gameState.getLegalActions(0):

            val = expectimax(1, 0, gameState.generateSuccessor(0, agentState))
            if val > maximum:
                maximum = val
                action = agentState
        
        # Final return of the action
        return action 
        #util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    # We use what be see in evaluationFunction
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    # Food section-----------
    
    # First, the food implemented on a list with the minimum distance and actual 
    # position of the food
    foodDistance = []
    minFoodDist = 0
    foodPos = newFood.asList()
    
    # We will pass trhought all the food and get their distance wiht pacman
    for food in foodPos:
        distances = util.manhattanDistance(newPos, food)
        foodDistance.append(distances)
    # If the list is empty, then we add the lower distance from foodDistance
    if len(foodDistance) > 0:
        minFoodDist = min(foodDistance)

    # Ghosts section--------
    
    # List of distances with the ghosts, minimun distance and actual position of ghosts
    ghostsDistance = []
    minGhostDist = 0
    ghostPos = currentGameState.getGhostPositions()
    # We well use also a varaible for index min ghost
    indexGhost = -1
    
    # We will pass trhought all the ghost and get their distance wiht pacman
    for ghost in ghostPos:
        distance = util.manhattanDistance(newPos, ghost)
        ghostsDistance.append(distance)
    # If the list is empty, then we add the lower distance from ghostsDistance
    # and we save the distance
    if len(ghostsDistance) > 0:
        minGhostDist = min(ghostsDistance)
        indexGhost = ghostsDistance.index(minGhostDist)

    # If there is no ghost then the score will depend on the food
    if (newScaredTimes[indexGhost] == -1):
        return currentGameState.getScore() - (minFoodDist)
    # If there is a ghost available to eat with the minimum distance, you gave to get it (a lot of points)
    elif (newScaredTimes[indexGhost] > 0):
        return currentGameState.getScore() + (100.0 / (minGhostDist + 1.0)) - (minFoodDist)
    # If there is ghosts
    else:
        return currentGameState.getScore() - (50.0 / (minGhostDist + 1.0)) - (minFoodDist)
    # util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
