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
        closestFood = float('+Inf')
        closestGhost = float('+Inf')

        food = currentGameState.getFood()
        score = successorGameState.getScore()
        ghostPositions = successorGameState.getGhostPositions()
        foodList = food.asList()
        newFoodList = newFood.asList()
        addScore = 0

        if newPos in foodList:
            addScore += 10.0

        distanceFromFood = [manhattanDistance(newPos, foodPosition) for foodPosition in newFoodList]
        availableFood = len(newFoodList)

        if len(distanceFromFood):
            closestFood = min(distanceFromFood)
        score += 10.0 / closestFood  - 4.0 * availableFood + addScore

        for ghostPosition in ghostPositions:
            distanceFromGhost = manhattanDistance(newPos, ghostPosition)
        closestGhost = min([closestGhost, distanceFromGhost])

        if closestGhost < 2:
            score -= 50.0

        return score


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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        result = self.max_value(gameState, 0, 0)[1]
        return result

    def is_terminal_state(self, gameState, depth, index):

        if gameState.isWin():
            return gameState.isWin()

        elif gameState.isLose():
            return gameState.isLose()

        elif gameState.getLegalActions(index) is 0:
            return gameState.getLegalActions(index)

        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth
    
    def max_value(self, gameState, depth, index):

        val = (float('-Inf'), None)
        legalActions = gameState.getLegalActions(index)

        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            numAgents = gameState.getNumAgents()
            expand = depth + 1
            currentPlayer = (depth + 1) % numAgents
            val = max([val, (self.value(successor, expand, currentPlayer), action)], key=lambda idx: idx[0])
        return val

    def min_value(self, gameState, depth, index):
        
        val = (float('+Inf'), None)
        legalActions = gameState.getLegalActions(index)

        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            numAgents = gameState.getNumAgents()
            expand = depth + 1
            currentPlayer = (depth + 1) % numAgents
            val = min([val, (self.value(successor, expand, currentPlayer), action)], key=lambda idx: idx[0])
        return val

    def value(self, gameState, depth, index):

        if self.is_terminal_state(gameState, depth, index):
            return self.evaluationFunction(gameState)

        elif index is 0:
            return self.max_value(gameState, depth, index)[0]

        else:
            return self.min_value(gameState, depth, index)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        alpha = float('-Inf')
        beta = float('+Inf')

        result = self.max_value(gameState, 0, 0, alpha, beta)
        return result[1]

    def is_terminal_state(self, gameState, depth, index):
        
        if gameState.isWin():
            return gameState.isWin()

        elif gameState.isLose():
            return gameState.isLose()

        elif gameState.getLegalActions(index) is 0:
            return gameState.getLegalActions(index)

        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth

    def max_value(self, gameState, depth, index, alpha, beta):

        val = (float('-Inf'), None)
        legalActions = gameState.getLegalActions(index)

        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            numAgents = gameState.getNumAgents()
            expand = depth + 1
            currentPlayer = expand % numAgents
            val = max([val, (self.value(successor, expand, currentPlayer, alpha, beta), action)], key=lambda idx: idx[0])
            if val[0] > beta:
                return val
            alpha = max(alpha, val[0])

        return val

    def min_value(self, gameState, depth, index, alpha, beta):

        val = (float('+Inf'), None)
        legalActions = gameState.getLegalActions(index)

        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            numAgents = gameState.getNumAgents()
            expand = depth + 1
            currentPlayer = expand % numAgents
            val = min([val, (self.value(successor, expand, currentPlayer, alpha, beta), action)], key=lambda idx: idx[0])
            if val[0] < alpha:
                return val
            beta = min(beta, val[0])

        return val
    
    def value(self, gameState, depth, index, alpha, beta):

        if self.is_terminal_state(gameState, depth, index):
            return self.evaluationFunction(gameState)

        elif index is 0:
            return self.max_value(gameState, depth, index, alpha, beta)[0]

        else:
            return self.min_value(gameState, depth, index, alpha, beta)[0]

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

        result = self.max_value(gameState, 0, 0)[1]
        return result
    
    def is_terminal_state(self, gameState, depth, index):

        if gameState.isWin():
            return gameState.isWin()

        elif gameState.isLose():
            return gameState.isLose()

        elif gameState.getLegalActions(index) is 0:
            return gameState.getLegalActions(index)

        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth

    def max_value(self, gameState, depth, index):

        val = (float('-Inf'), None)
        legalActions = gameState.getLegalActions(index)

        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            numAgents = gameState.getNumAgents()
            expand = depth + 1
            currentPlayer = expand  % numAgents
            val = max([val, (self.value(successor, expand, currentPlayer), action)], key=lambda idx: idx[0])
        return val
    
    def expected_value(self, gameState, depth, index):
        val = list()
        legalActions = gameState.getLegalActions(index)

        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            numAgents = gameState.getNumAgents()
            expand = depth + 1
            currentPlayer = expand % numAgents
            val.append(self.value(successor, expand, currentPlayer))
        expectedVal = sum(val) / len(val)
        return expectedVal

    def value(self, gameState, depth, index):
        if self.is_terminal_state(gameState, depth, index):
            return self.evaluationFunction(gameState)

        elif index is 0:
            return self.max_value(gameState, depth, index)[0]

        else:
            return self.expected_value(gameState, depth, index)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    scaredGhosts = list()
    enemyGhosts = list()
    enemyGhostPositions = list()
    scaredGhostsPositions = list()

    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    capsulesPositions = currentGameState.getCapsules()
    ghostPositions = currentGameState.getGhostPositions()

    ghostStates = currentGameState.getGhostStates()
    scaredGhostsTimer = [ghostState.scaredTimer for ghostState in ghostStates]
    remainingFood = len(foodPositions)
    remainingCapsules = len(capsulesPositions)
    score = currentGameState.getScore()

    closestFood = float('+Inf')
    closestEnemyGhost = float('+Inf')
    closestScaredGhost = float('+Inf')

    distanceFromFood = [manhattanDistance(pacmanPosition, foodPosition) for foodPosition in foodPositions]
    if len(distanceFromFood) is not 0:
        closestFood = min(distanceFromFood)
        score -= 1.0 * closestFood

    for ghost in ghostStates:
        if ghost.scaredTimer is not 0:
            enemyGhosts.append(ghost)
        else:
            scaredGhosts.append(ghost)

    for enemyGhost in enemyGhosts:
        enemyGhostPositions.append(enemyGhost.getPosition())

    if len(enemyGhostPositions) is not 0:
        distanceFromEnemyGhost = [manhattanDistance(pacmanPosition, enemyGhostPosition) for enemyGhostPosition in enemyGhostPositions]
        closestEnemyGhost = min(distanceFromEnemyGhost)
        score -= 2.0 * (1 / closestEnemyGhost)

    for scaredGhost in scaredGhosts:
        scaredGhostsPositions.append(scaredGhost.getPosition())

    if len(scaredGhostsPositions) is not 0:
        distanceFromScaredGhost = [manhattanDistance(pacmanPosition, scaredGhostPosition) for scaredGhostPosition in scaredGhostsPositions]
        closestScaredGhost = min(distanceFromScaredGhost)
        score -= 3.0 * closestScaredGhost

    score -= 20.0 * remainingCapsules
    score -= 4.0 * remainingFood
    return score


# Abbreviation
better = betterEvaluationFunction
