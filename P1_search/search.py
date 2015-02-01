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

    #1. initilize start node
    parent = None
    action = None
    cost = 0
    startState = problem.getStartState()
    node = Node(parent,action,cost,startState)


    #2. check if is goal state
    if problem.isGoalState(node.state):
        return node.path()

    #3. else, initialize the frontier with the start node
    frontier = util.Stack()
    frontierNodes = {}
    frontier.push(startState)
    frontierNodes[startState] = node
    #print "Push:",node.state

    #4. initilized the explroed set (a set of state)
    explored = []

    while True:
        if frontier.isEmpty():
            return None # failuer
        state = frontier.pop()
        node = frontierNodes.pop(state, None)
        #print "Pop:",node.state

        if problem.isGoalState(state):
            return node.path()

        explored.append(state)

        for successor,action,stepCost in problem.getSuccessors(state):
            child = Node(node,action,node.cost+stepCost,successor)
            if child.state not in explored:
                if frontierNodes.has_key(child.state):
                    # update node with different path                    
                    frontierNodes[child.state] = child
                else: # not in frontier
                    frontier.push(child.state)
                    frontierNodes[child.state] = child
    
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #1. initilize start node
    parent = None
    action = None
    cost = 0
    startState = problem.getStartState()
    node = Node(parent,action,cost,startState)

    #2. check if is goal state
    if problem.isGoalState(node.state):
        return node.path()

    #3. else, initialize the frontier with the start node
    frontier = util.Queue()
    frontierNodes = {}
    frontier.push(startState)
    frontierNodes[startState] = node

    #4. initilized the explroed set (a set of state)
    explored = []

    while True:
        if frontier.isEmpty():
            return None # failuer
        state = frontier.pop()
        node = frontierNodes.pop(state, None)
        #print "Pop:",node.state

        if problem.isGoalState(state):
            return node.path()

        explored.append(state)

        for successor,action,stepCost in problem.getSuccessors(state):
            child = Node(node,action,node.cost+stepCost,successor)
            if child.state not in explored and not frontierNodes.has_key(child.state):
                frontier.push(child.state)
                frontierNodes[child.state] = child
                #print "Push the children: "
                #print child.state

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #1. initilize start node

    parent = None
    action = None
    cost = 0
    startState = problem.getStartState()
    node = Node(parent,action,cost,startState)


    #2. check if is goal state
    if problem.isGoalState(node.state):
        return node.path()

    #3. else, initialize the frontier with the start node
    frontier = util.PriorityQueue()
    frontierNodes = {}
    frontier.push(startState,cost)
    frontierNodes[startState] = node
    print "Push:",node.state
    
    #4. initilized the explroed set (a set of state)
    explored = []

    while True:
        if frontier.isEmpty():
            return None # failure
        state = frontier.pop() # pop the state with lowest cost
        node = frontierNodes.pop(state, None)
        #print "Pop:",node.state
        

        if problem.isGoalState(state):
            return node.path()

        explored.append(state)

        for successor,action,stepCost in problem.getSuccessors(state): 
            child = Node(node,action,node.cost+stepCost,successor)
            if child.state not in explored and not frontierNodes.has_key(child.state):
                # not in frontier
                frontier.push(child.state,child.cost)
                frontierNodes[child.state] = child
                #print "Push",child.state
            elif frontierNodes.has_key(child.state):
                old_cost = frontierNodes[child.state].cost
                if old_cost > child.cost:
                    frontierNodes[child.state] = child
                    #ADD CODE HERE
                    #also need to update the priority in frontier, or you might be wrong even
                    #you can pass all test cases
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #1. initilize start node

    parent = None
    action = None
    startState = problem.getStartState()
    g = 0   
    h = heuristic(startState,problem)
    f = g+h
    node = Node(parent,action,g,startState)


    #2. check if is goal state
    if problem.isGoalState(node.state):
        return node.path()

    #3. else, initialize the frontier with the start node
    frontier = util.PriorityQueue()
    frontierNodes = {}
    frontier.push(startState,f)
    frontierNodes[startState] = [node,f]
    print "Push",node.state,g,h
    
    #4. initilized the explroed set (a set of state)
    explored = []

    while True:
        if frontier.isEmpty():
            return None # failuer
        state = frontier.pop() # pop the state with lowest cost
        node = frontierNodes.pop(state, None)[0]
        #print "Pop",node.state
        

        if problem.isGoalState(state):
            return node.path()

        explored.append(state)

        for successor,action,stepCost in problem.getSuccessors(state):         
            child = Node(node,action,node.cost+stepCost,successor)
            g = child.cost
            h = heuristic(successor,problem)
            f = g+h
            if child.state not in explored and not frontierNodes.has_key(child.state):
                # not in frontier
                frontier.push(child.state,f)
                frontierNodes[child.state] = [child,f]
                #print "Push",child.state
            elif frontierNodes.has_key(child.state):
                old_f = frontierNodes[child.state][1]
                if old_f > f:
                    frontierNodes[child.state] = [child,f]
    

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
