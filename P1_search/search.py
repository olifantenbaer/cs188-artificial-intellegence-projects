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
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #define the priority function (return f=g+h)
    def priorityFunction(node):
        if len(node.path()) == 0:
            return float("inf")
        else:
            return 1.0/len(node.path())

    return geneticSearch(problem,priorityFunction)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #define the priority function (return f=g+h)
    def priorityFunction(node):
        return len(node.path())

    return geneticSearch(problem,priorityFunction)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return aStarSearch(problem)
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #define the priority function (return f=g+h)
    def priorityFunction(node):
        return node.cost+heuristic(node.state,problem)

    return geneticSearch(problem,priorityFunction,heuristic)
    
def geneticSearch(problem,priorityFunction,heuristic=nullHeuristic):
    #1. initilize the start node
    parent = None
    action = None
    pathCost = 0
    startState = problem.getStartState()
    node = Node(parent,action,pathCost,startState)

    #2. check if is goal state
    if problem.isGoalState(node.state):
        return node.path() 

    #3. initialize the frontier with the start node
    frontier = util.PriorityQueueWithFunction(priorityFunction)
    frontier.push(node)
    frontierStates = {}
    frontierStates[startState] = priorityFunction(node)
    
    #4. initilized the explroed set (a set of state)
    explored = []

    while True:
        if frontier.isEmpty():
            return None # failure

        node = frontier.pop()
        if frontierStates.pop(node.state,None) == None:
            continue

        if problem.isGoalState(node.state):
            return node.path()

        explored.append(node.state)

        for successor,action,stepCost in problem.getSuccessors(node.state):         
            child = Node(node,action,node.cost+stepCost,successor)
            f = priorityFunction(child)
            if child.state not in explored:
                frontier.push(child)
                frontierStates[child.state] = f
            '''if child.state not in explored and not frontierStates.has_key(child.state):
                frontier.push(child)
                frontierStates[child.state] = f
            elif frontierStates.has_key(child.state):
                old_f = frontierStates[child.state]
                if old_f > f:# if in frontier with higher f, update
                    frontier.push(child)
                    frontierStates[child.state] = f'''



class Node:
    def __init__(self,parent,action,cost,state):
        self.parent = parent
        #self.length = length
        self.action = action # the action which taken at parent state
        self.cost = cost
        self.state = state
    '''return the solution (a list of actions)'''
    def path(self):
        if self.parent == None:
            return []
        else:
            return self.parent.path() + [self.action]



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
