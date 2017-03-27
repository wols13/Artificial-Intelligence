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
import datetime

from game import Agent

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

        return self.depthFirstMiniMax(gameState, 0, 1)[1]

    def depthFirstMiniMax(self, gameState, agent, level):
        utility = 0
        bestAction = Directions.STOP

        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), bestAction

        # Apply Player's moves to get successor states
        legalActions = gameState.getLegalActions(agent)

        if gameState.getNumAgents() * self.depth == level:
            utility = float('inf')
            for i in range(len(legalActions)):
                temp = self.evaluationFunction(gameState.generateSuccessor(agent, legalActions[i]))
                if temp < utility:
                    utility = temp
                    bestAction = legalActions[i]
        elif agent != 0:
            utility = float('inf')
            for i in range(len(legalActions)):
                temp = self.depthFirstMiniMax(gameState.generateSuccessor(agent, legalActions[i]), (agent + 1) % gameState.getNumAgents(), level + 1)[0]
                if temp < utility:
                    utility = temp
                    bestAction = legalActions[i]
        else:
            utility = -float('inf')
            for i in range(len(legalActions)):
                temp = self.depthFirstMiniMax(gameState.generateSuccessor(agent, legalActions[i]), agent + 1, level + 1)[0]
                if temp > utility:
                    utility = temp
                    bestAction = legalActions[i]
        return utility, bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        return self.alphaBeta(gameState, 0, 1, -float('inf'), float('inf'))[1]

    def alphaBeta(self, gameState, agent, level, alpha, beta):
        utility = 0
        currentAlpha = alpha
        currentBeta = beta
        bestAction = Directions.STOP

        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), bestAction

        legalActions = gameState.getLegalActions(agent)

        if gameState.getNumAgents() * self.depth == level:
            for i in range(len(legalActions)):
                temp = self.evaluationFunction(gameState.generateSuccessor(agent, legalActions[i]))
                if temp < currentBeta:
                    currentBeta = temp
                    bestAction = legalActions[i]
                if currentBeta <= currentAlpha:
                    break
            utility = currentBeta
        elif agent != 0:
            for i in range(len(legalActions)):
                temp = self.alphaBeta(gameState.generateSuccessor(agent, legalActions[i]), (agent + 1) % gameState.getNumAgents(), level + 1, currentAlpha, currentBeta)[0]
                if temp < currentBeta:
                    currentBeta = temp
                    bestAction = legalActions[i]
                if currentBeta <= currentAlpha:
                    break
            utility = currentBeta
        else:
            for i in range(len(legalActions)):
                temp = self.alphaBeta(gameState.generateSuccessor(agent, legalActions[i]), agent + 1, level + 1, currentAlpha, currentBeta)[0]
                if temp > currentAlpha:
                    currentAlpha = temp
                    bestAction = legalActions[i]
                if currentBeta <= currentAlpha:
                    break
            utility = currentAlpha
        return utility, bestAction


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
        return self.expectiMax(gameState, 0, 1)[1]

    def expectiMax(self, gameState, agent, level):
        utility = 0
        bestAction = Directions.STOP

        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), bestAction

        # Apply Player's moves to get successor states
        legalActions = gameState.getLegalActions(agent)

        if gameState.getNumAgents() * self.depth == level:
            for i in range(len(legalActions)):
                utility += self.evaluationFunction(gameState.generateSuccessor(agent, legalActions[i]))
            utility /= len(legalActions)
        elif agent != 0:
            for i in range(len(legalActions)):
                utility += self.expectiMax(gameState.generateSuccessor(agent, legalActions[i]), (agent + 1) % gameState.getNumAgents(), level + 1)[0]
            utility /= len(legalActions)
        else:
            utility = -float('inf')
            for i in range(len(legalActions)):
                temp = self.expectiMax(gameState.generateSuccessor(agent, legalActions[i]), agent + 1, level + 1)[0]
                if temp > utility:
                    utility = temp
                    bestAction = legalActions[i]
        return utility, bestAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    # States that result in a loss evaluate to -infinity
    if currentGameState.isLose():
        return -float('inf')

    # If the state results in a win, the final score is a valid state evaluation
    if currentGameState.isWin():
        return currentGameState.getScore()

    # For non-terminal states, evaluation = (current score) - (closest food to pacman)
    pacmanPosition = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    closestFood = float('inf')

    for x in range(foodGrid.width):
        for y in range(foodGrid.height):
            if foodGrid[x][y]:
                closestFood = min(abs(pacmanPosition[0] - x) + abs(pacmanPosition[1] - y), closestFood)

    return currentGameState.getScore() - closestFood


# Abbreviation
better = betterEvaluationFunction


# add the function scoreEvaluationFunction to multiAgents.py
def scoreEvaluationFunction(currentGameState):
   """
     This default evaluation function just returns the score of the state.
     The score is the same one displayed in the Pacman GUI.

     This evaluation function is meant for use with adversarial search agents
   """
   return currentGameState.getScore()


# add this class to multiAgents.py
# the following class corrects and replaces the previous MonteCarloAgent class released on March 19
# the only differences between this version, and the one released on March 19 are:
#       * line 37 of this file, "if self.Q" has been replaced by "if Q"
#       * line 45 of this file, where "assert( Q == 'contestClassic' )" has been added
class MonteCarloAgent(MultiAgentSearchAgent):
    """
        Your monte-carlo agent (question 5)
        ***UCT = MCTS + UBC1***
        TODO:
        1) Complete getAction to return the best action based on UCT.
        2) Complete runSimulation to simulate moves using UCT.
        3) Complete final, which updates the value of each of the states visited during a play of the game.

        * If you want to add more functions to further modularize your implementation, feel free to.
        * Make sure that your dictionaries are implemented in the following way:
            -> Keys are game states.
            -> Value are integers. When performing division (i.e. wins/plays) don't forget to convert to float.
      """

    def __init__(self, evalFn='mctsEvalFunction', depth='-1', timeout='40', numTraining=100, C='2', Q=None):
        # This is where you set C, the depth, and the evaluation function for the section "Enhancements for MCTS agent".
        if Q:
            if Q == 'minimaxClassic':
                pass
            elif Q == 'testClassic':
                pass
            elif Q == 'smallClassic':
                pass
            else: # Q == 'contestClassic'
                assert( Q == 'contestClassic' )
                pass
        # Otherwise, your agent will default to these values.
        else:
            self.C = int(C)
            # If using depth-limited UCT, need to set a heuristic evaluation function.
            if int(depth) > 0:
                evalFn = 'scoreEvaluationFunction'
        self.states = []
        self.plays = dict()
        self.wins = dict()
        self.calculation_time = datetime.timedelta(milliseconds=int(timeout))

        self.numTraining = numTraining

        "*** YOUR CODE HERE ***"

        MultiAgentSearchAgent.__init__(self, evalFn, depth)

    def update(self, state):
        """
        You do not need to modify this function. This function is called every time an agent makes a move.
        """
        self.states.append(state)

    def getAction(self, gameState):
        """
        Returns the best action using UCT. Calls runSimulation to update nodes
        in its wins and plays dictionary, and returns best successor of gameState.
        """
        "*** YOUR CODE HERE ***"
        games = 0
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            games += 1

        util.raiseNotDefined()

    def run_simulation(self, state):
        """
        Simulates moves based on MCTS.
        1) (Selection) While not at a leaf node, traverse tree using UCB1.
        2) (Expansion) When reach a leaf node, expand.
        4) (Simulation) Select random moves until terminal state is reached.
        3) (Backpropapgation) Update all nodes visited in search tree with appropriate values.
        * Remember to limit the depth of the search only in the expansion phase!
        Updates values of appropriate states in search with with evaluation function.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        """
        Called by Pacman game at the terminal state.
        Updates search tree values of states that were visited during an actual game of pacman.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def mctsEvalFunction(state):
    """
    Evaluates state reached at the end of the expansion phase.
    """
    return 1 if state.isWin() else 0
