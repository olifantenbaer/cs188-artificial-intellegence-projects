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
        currentFood = currentGameState.getFood()
        currentFoodList = currentFood.asList()
        newFoodList = newFood.asList()
        foodDistance = float('inf')

        for food in newFoodList:
            distance1 = manhattanDistance(food, newPos)
            foodDistance = min(distance1,foodDistance)

        if len(newFoodList) < len(currentFoodList):
            foodDistance = 0.000000001


        newGhostPosition = successorGameState.getGhostPositions()
        ghostDistance = float('inf')

        for ghost in newGhostPosition:
            distance2 = manhattanDistance(ghost, newPos)
            ghostDistance = min(distance2, ghostDistance)

        if ghostDistance < 2:
            return -float('inf')
        else:
            ghostDistance = 0

        return 1.0/foodDistance + ghostDistance

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
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
        legalMoves = gameState.getLegalActions(self.index)
        states = [gameState.generateSuccessor(self.index, action) for action in legalMoves]
        # print self.Minimax(self.index,gameState,0)
        # exit()
        scores = [self.Minimax(self.index+1,state,0) for state in states]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]
    
    def Minimax(self,agentIndex,gameState,currentDepth):
        if currentDepth == self.depth:
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agentIndex)

        if len(legalMoves) == 0:
            return self.evaluationFunction(gameState)

        num = gameState.getNumAgents()

        if agentIndex == num - 1:
            nextAgentIndex = 0
            currentDepth += 1
        else:
            nextAgentIndex = agentIndex + 1

        if agentIndex == self.index:
            return max([self.Minimax(nextAgentIndex,gameState.generateSuccessor(agentIndex, action),currentDepth) for action in legalMoves])
        else:
            return min([self.Minimax(nextAgentIndex,gameState.generateSuccessor(agentIndex, action),currentDepth) for action in legalMoves])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState,-float('inf'),float('inf'),0)[1]

    def maxValue(self,gameState,alpha,beta,currentDepth):
        if currentDepth == self.depth:
            foo= (self.evaluationFunction(gameState),None)
            return foo

        legalMoves = gameState.getLegalActions(self.index)

        if len(legalMoves) == 0:
            foo= (self.evaluationFunction(gameState),None)
            return foo

        v = (-float('inf'),None)
        for action in legalMoves:
            v1=(self.minValue(gameState.generateSuccessor(self.index,action),alpha,beta,currentDepth,self.index+1)[0],action)
            v = max(v,v1,key=lambda x:x[0])
            if v[0]>= beta: return v
            alpha = max(alpha,v[0])
        return v

    def minValue(self,gameState,alpha,beta,currentDepth,agentIndex):
        
        if currentDepth == self.depth:
            return (self.evaluationFunction(gameState),None)
  
        legalMoves = gameState.getLegalActions(agentIndex)

        if len(legalMoves) == 0:
            foo= (self.evaluationFunction(gameState),None)
            return foo

        num = gameState.getNumAgents()
        v = (float('inf'),None)
        if agentIndex == num - 1:
            nextAgentIndex = 0
            currentDepth += 1
            
            for action in legalMoves:
                v1=(self.maxValue(gameState.generateSuccessor(agentIndex,action),alpha,beta,currentDepth)[0],action)
                v = min(v,v1,key=lambda x:x[0])
                if v[0]<= alpha: 
                    return v
                beta = min(beta,v[0])
            return v
        else:
            nextAgentIndex = agentIndex + 1
            for action in legalMoves:
                v1 = (self.minValue(gameState.generateSuccessor(agentIndex,action),alpha,beta,currentDepth,nextAgentIndex)[0],action)
                v = min(v,v1,key=lambda x:x[0])
                if v[0]<= alpha: return v
                beta = min(beta,v[0])
            return v

        


        




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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
