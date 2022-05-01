# qlearningAgents.py
# ------------------
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

#Antonio Pintado u172771 and Amanda Pintado u137702


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qvalue = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        tState = 'TERMINAL_STATE'
        # If the state is terminal or there is no Qvalue stored already then return 0.0,
        # if not return the value stored in the dictionary
        if state == tState or (state, action) not in self.qvalue:
            return 0.0
        # Call to values
        return self.qvalue[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actionV = float('-inf')
        # Like in the value iteration, we compute the best Q value for all the
        # possible actions, first initializing a minus infinite value in action
        # V to obtain the better value for the actions available
        # Here we will get our (N,S,E,W) legal actions to move
        for action in self.getLegalActions(state):
            expectedQVal = self.getQValue(state, action)
            if actionV < expectedQVal:
                actionV = expectedQVal

        if actionV == float('-inf'):
            # Return 0.0 if its terminal (no possible actions)
            return 0.0

        return actionV

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        # First check, terminal state is NONE
        if not legalActions:
            return None
        # Save first action and value to be able to iterate in the for
        maxAction = legalActions[0]
        maxValue = self.qvalue[(state, maxAction)]
        # We perform like in computeValueFromQValues, but
        # in this case we compute the argmax of the possible actions, instead of the max of the values
        # Check the legal actions and save them
        for action in legalActions:
            aux = self.qvalue[(state, action)]
            if aux > maxValue:
                maxAction = action
                maxValue = aux
        # Just get the best action
        return maxAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # Random Prob. p = true , 1-p= false
        if util.flipCoin(self.epsilon):
            # In the case the coin its true we choose the action randomly (with random.choice(list), in this case
            # list is the list of the legal Actions)
            return random.choice(legalActions)
        # If the coin its false then return the current best action
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # First get the new info (Qvalue)
        q = self.getQValue(state, action)
        # Then update again the Qvalue with an a running average (as we saw in class)
        self.qvalue[(state, action)] = (1 - self.alpha) * q + self.alpha * (reward + self.discount * self.getValue(nextState))


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Init the total val and get the features for the sum
        val = 0
        # Like in the formula we compute the Qvalue with the summation of the weights
        # times the feature of the (state, action)
        for key in self.featExtractor.getFeatures(state, action).keys():
            val += self.weights[key] * self.featExtractor.getFeatures(state, action)[key]
        return val

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # First get the difference (does not change in the for), with the reward,
        # discount, value and the Qvalue
        upd = (reward + (self.discount * self.getValue(nextState))) - self.getQValue(state, action)
        # Then update the weights from the old weight plus alpha multiplied by the
        # difference and the feature of him
        # To make the sum
        for key in self.featExtractor.getFeatures(state, action).keys():
            self.weights[key] = self.weights[key] + self.alpha * upd * self.featExtractor.getFeatures(state, action)[key]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
