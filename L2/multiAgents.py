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


from game import Agent
from game import Directions
import random
import util
from util import manhattanDistance

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        
        #First, the food-------------
        #List for food, minimum distance and actual position of the food
        foodDistance = []
        minFoodDist = 0
        foodPos = newFood.asList()
        
        #We will pass trhought all the food and get their distance wiht pacman
        for food in foodPos:
            distances = util.manhattanDistance(newPos, food)
            foodDistance.append(distances)

        #if the list is empty, then we add the lower distance from foodDistance
        if len(foodDistance) > 0:
            minFoodDist = min(foodDistance)
        
        #For the ghosts--------
        #List of distances with the ghosts, minimun distance and actual position of ghosts
        ghostsDistance = []
        minGhostDist = 0
        ghostPos = currentGameState.getGhostPositions()
        
        #We will pass trhought all the ghost and get their distance wiht pacman
        for ghost in ghostPos:
            distance = util.manhattanDistance(newPos, ghost)
            ghostsDistance.append(distance)
        
        #if the list is empty, then we add the lower distance from ghostsDistance
        if len(ghostsDistance) > 0:
            minGhostDist = min(ghostsDistance)

        #The score will be determined by the ghost and the food missed
        return successorGameState.getScore()-(50.0 / (minGhostDist + 1.0))-(minFoodDist)

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
        self.index = 0 # Pacman is always agent index 0
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
        """
        "*** YOUR CODE HERE ***"
        
        def minimax(agent, depth, gameState):
         
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  
                #devuelve la evaluacion en caso de perder/ganar/o llegar a la profundidad definida
                return self.evaluationFunction(gameState)
            
            if agent == 0:  
                #Maximiza para Pacman  
                maxim = max(minimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
                return maxim
            
            else:  
                #Minimiza para fantasmas
                nextAgent = agent + 1  
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth = depth + 1#aumentamos la profundidad
                #Calculamos el valor minimo de los nodos 
                minim = min(minimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
                return minim                      
               
      
        maximum = float("-inf") #Inicio raiz valor infinito negativo
        action = Directions.WEST
    
       
        for agentState in gameState.getLegalActions(0):#acciones
        
            suc = gameState.generateSuccessor(0, agentState)
        
            val = minimax(1, 0, suc)
          
            if val >= maximum:
                maximum = val
                action = agentState
      
        return action
        
        
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        "*** YOUR CODE HERE ***"
        
        #Funcion que maximiza
        def maximizer(agent, depth, game_state, alpha, beta): 
            score = float("-inf")
            
            for newState in game_state.getLegalActions(agent):
                
                suc = game_state.generateSuccessor(agent, newState)
                score = max(score, alphaBetaPrune(1, depth, suc, alpha, beta))
                if score > beta:
                    return score
                alpha = max(alpha, score)
           
            return score
        #Funcion que minimiza
        def minimizer(agent, depth, game_state, alpha, beta):  
            score = float("inf")

            next_agent = agent + 1  
            if game_state.getNumAgents() == next_agent:
                next_agent = 0
            if next_agent == 0:
                depth = depth + 1 #aumentamos la profundidad

            for newState in game_state.getLegalActions(agent):
                
                suc = game_state.generateSuccessor(agent, newState)
                
                score = min(score, alphaBetaPrune(next_agent, depth, suc, alpha, beta))
                
                if score < alpha:
                    return score
                beta = min(beta, score)
                
            return score
        
        def alphaBetaPrune(agent, depth, game_state, alpha, beta):
            
            if game_state.isLose() or game_state.isWin() or depth == self.depth:  
                #devuelve la evaluacion en caso de perder/ganar/o llegar a la profundidad definida
                return self.evaluationFunction(game_state)

            if agent == 0:
                #Maximiza para Pacman    
                return maximizer(agent, depth, game_state, alpha, beta)
            else: 
                #Minimiza para fantasmas
                return minimizer(agent, depth, game_state, alpha, beta)

       
        action = Directions.WEST
        val = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for agentState in gameState.getLegalActions(0):
            
            suc = gameState.generateSuccessor(0, agentState)      
            ghostVal = alphaBetaPrune(1, 0, suc, alpha, beta)
          
            if ghostVal > val:
                val = ghostVal
                action = agentState
                
            if val > beta:
                return val
            
            alpha = max(alpha, val)
            
            
        return action
        #util.raiseNotDefined()

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
            
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  
                return self.evaluationFunction(gameState)
            
            if agent == 0:
                #Maximiza para Pacman
                return max(expectimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            else: 
                #No toma el minimo sino la expectativa
                nextAgent = agent + 1  # Calcula el siguiente agente
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth = depth + 1#aumentamos la profundidad
                return sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent)) / float(len(gameState.getLegalActions(agent)))

        
        maximum = float("-inf")
        action = Directions.WEST
        
        for agentState in gameState.getLegalActions(0):
          
            val = expectimax(1, 0, gameState.generateSuccessor(0, agentState))
                     
            if val > maximum:
                maximum = val
                action = agentState
        
      
        return action #devuelve la accion
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <?>
    """
    
    "*** YOUR CODE HERE ***"
    
    #We used what be see in evaluationFunction
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    #First, the food-------------
    #List for food, minimum distance and actual position of the food
    foodDistance = []
    minFoodDist = 0
    foodPos = newFood.asList()
    
    #We will pass trhought all the food and get their distance wiht pacman
    for food in foodPos:
        distances = util.manhattanDistance(newPos, food)
        foodDistance.append(distances)

    #if the list is empty, then we add the lower distance from foodDistance
    if len(foodDistance) > 0:
        minFoodDist = min(foodDistance)

    #For the ghosts--------
    #List of distances with the ghosts, minimun distance and actual position of ghosts
    ghostsDistance = []
    minGhostDist = 0
    ghostPos = currentGameState.getGhostPositions()    
    #We well use also a varaible for index min ghost
    indexGhost = -1
    
    #We will pass trhought all the ghost and get their distance wiht pacman
    for ghost in ghostPos:
        distance = util.manhattanDistance(newPos, ghost)
        ghostsDistance.append(distance)
        
    #if the list is empty, then we add the lower distance from ghostsDistance 
    #and we save the distance
    if len(ghostsDistance) > 0:
        minGhostDist = min(ghostsDistance)
        indexGhost = ghostsDistance.index(minGhostDist)
            
    #If there is no ghost then the score will depend on the food
    if(newScaredTimes[indexGhost] == -1):
        return currentGameState.getScore()-(minFoodDist)
    #If there is a ghost available to eat with the minimum distance, you gave to get it (a lot of points)
    elif(newScaredTimes[indexGhost] > 0):
        return currentGameState.getScore() + (100.0 / (minGhostDist + 1.0))-(minFoodDist)
    #If there is ghosts
    else:
        return currentGameState.getScore()-(50.0 / (minGhostDist + 1.0))-(minFoodDist)
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

