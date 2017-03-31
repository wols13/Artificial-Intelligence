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
import math
import random

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

      DESCRIPTION: States that result in a loss, evaluate to -infinity
                   States that result in a win, the final score is a valid state evaluation
                   Non-terminal states, evaluation = (current score) - (closest food to pacman)
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

    # farthestGhost = 0
    # for ghost in currentGameState.getGhostPositions(0):
    #     farthestGhost = max(abs(pacmanPosition[0] - ghost[0]) + abs(pacmanPosition[1] - ghost[1]), farthestGhost)

    return currentGameState.getScore() - (2 * closestFood) #+ farthestGhost


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
                depth = 8
                self.C = 4
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

    def getUcbValue(self, gameState, parent_visits):
        if gameState not in self.plays or self.plays[gameState] == 0:
            return float('inf')
        else:
            value_estimate = float(self.wins[gameState]) / self.plays[gameState]
            result = math.sqrt(math.log(parent_visits) / self.plays[gameState])
            return value_estimate + (self.C * result)

    def getAction(self, gameState):
        """
        Returns the best action using UCT. Calls runSimulation to update nodes
        in its wins and plays dictionary, and returns best successor of gameState.
        """

        # Run UCT simulations while there is time
        games = 0
        begin = datetime.datetime.utcnow()

        if gameState.isWin() or gameState.isLose():
            return Directions.STOP

        if len(gameState.getLegalPacmanActions()) == 1:
            return gameState.getLegalPacmanActions()[0]

        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self.run_simulation(gameState)
            games += 1

        # Return best action given UCT simulation outcome
        best_action = None
        best_action_evaluation = -float('inf')
        for action in gameState.getLegalPacmanActions():
            successor = gameState.generatePacmanSuccessor(action)
            successor_value = 0
            if successor not in self.plays or self.plays[successor] == 0:
                successor_value = -float('inf')
            else:
                successor_value = float(self.wins[successor]) / self.plays[successor]
            if successor_value > best_action_evaluation:
                best_action_evaluation = successor_value
                best_action = action
        return best_action

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

        current_state = state
        current_agent = 0
        state_to_expand = None
        visited_states = dict()
        while state_to_expand is None:
            if current_state.isWin() or current_state.isLose():
                state_to_expand = current_state
                break
            # visited_states[current_state] = current_agent
            legal_actions = current_state.getLegalActions(current_agent)
            max_ucb_value = -float('inf')
            next_state = None

            parent_visits = 0
            for action in legal_actions:
                successor = current_state.generateSuccessor(current_agent, action)
                if successor in self.plays:
                    parent_visits += self.plays[successor]

            for action in legal_actions:
                successor = current_state.generateSuccessor(current_agent, action)
                if successor not in self.plays:
                    state_to_expand = successor
                    break
                else:
                    successor_ucb_value = self.getUcbValue(successor, parent_visits)
                    if successor_ucb_value > max_ucb_value:
                        max_ucb_value = successor_ucb_value
                        next_state = successor
            current_agent = (current_agent + 1) % state.getNumAgents()
            if state_to_expand is None:
                current_state = next_state
                visited_states[current_state] = current_agent
        visited_states[state_to_expand] = current_agent

        level = 1
        while True:
            if state_to_expand.isWin() or state_to_expand.isLose():
                break
            if self.depth != -1 and state_to_expand.getNumAgents() * self.depth == level:
                break

            legal_actions = state_to_expand.getLegalActions(current_agent)
            random_action_index = random.randint(0, len(legal_actions) - 1)
            state_to_expand = state_to_expand.generateSuccessor(current_agent, legal_actions[random_action_index])
            level += 1
            current_agent = (current_agent + 1) % state_to_expand.getNumAgents()

        value = self.evaluationFunction(state_to_expand)
        for node in visited_states:
            current_value = value
            if visited_states[node] != 0:
                current_value = value #########################
            if node in self.plays:
                self.plays[node] += 1
                self.wins[node] += current_value
            else:
                self.plays[node] = 1
                self.wins[node] = current_value

    def final(self, state):
        """
        Called by Pacman game at the terminal state.
        Updates search tree values of states that were visited during an actual game of pacman.
        """
        return True


def mctsEvalFunction(state):
    """
    Evaluates state reached at the end of the expansion phase.
    """
    return 1 if state.isWin() else 0
