# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import copy
import math

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if self.actions is not None and i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        #self.goal = goal
        self.goal = gameState.getFood().asList()[0]
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# Question 5 & 6: Corners Problem  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        #the four 'False's represent the four corners are not explored
        self.startState = (self.startingPosition,False,False,False,False)
        #print self.startingPosition # (1,2) - tuple

        

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        return self.startState

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        return state[1] and state[2] and state[3] and state[4]

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"
            x,y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            if not hitsWall:
                pos = (nextx,nexty)
                successor = map(lambda x:x,state)
                successor[0] = pos
                try:                
                    index = self.corners.index(pos) + 1
                    successor[index] = True
                except: # not one of the corners
                    pass

                successor = tuple(successor)
                successors.append((successor,action,1))


        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """

    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"

    "*** helper function ***"
    # Find the shortest manhattan cost by enumeration to solve the TSP problem
    # WARNING: Cannot deal with n-goal problems when n grows big enough
    def shortestManhattanCostThroughAllDestinations(startPosition,destPositions):
        if len(destPositions) == 0:
            return 0

        candidates = []
        for pos in destPositions:
            a = util.manhattanDistance( startPosition, pos )
            cpyDestPositions = copy.deepcopy(destPositions)
            cpyDestPositions.remove(pos)
            b = shortestManhattanCostThroughAllDestinations(pos,cpyDestPositions)
            candidates.append(a+b)
        return min(candidates)
    "*** END of helper function ***"


    startPosition = state[0]
    destPositions = []
    for index in range(len(corners)):
        if not state[index+1]:#if the corner is not explored
            destPositions.append(corners[index])
    return shortestManhattanCostThroughAllDestinations(startPosition,destPositions)

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

#####################################################
# Question 7 & 8: FoodSearchProblem  #
#####################################################


class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """

    # Experiment 1
    # 11470 nodes, optimal, 25.6 sec
    # return allManhattanHeuristic(state, problem)

    # Experiment 2
    # 10908 nodes, optimal, 22.3 sec
    # return NNManhattanHeuristic(state, problem)

    # Experiment 3
    # 7547 nodes, optimal, 13.1 sec
    # return MSTEuclideanHeuristic(state, problem)

    # Experiment 4
    # 7263 nodes, optimal, >100 sec
    # return RectilinearMinimumSteinerSpanningTreeHeuristic(state, problem)

    # Experiment 5
    # 250 nodes, optimal, 0.4 sec
    return MSTManhattanHeuristicWithWallPenaltyHeuristic(state, problem)


###########################################################################
# two simple heuristics
###########################################################################

#sum over food, distance to closest food/pacman
def allManhattanHeuristic(state, problem): 
    position, foodGrid = state
    foodPositions = foodGrid.asList()
    if len(foodPositions)==0:return 0

    allManhattan = 0
    allPositions = [position] + foodPositions
    
    for pos in foodPositions:
        minManhattan = 999999
        for pos2 in allPositions:
            dist = util.manhattanDistance( pos2, pos )
            if dist == 0: 
                continue
            if dist < minManhattan:
                minManhattan = dist
        allManhattan += minManhattan

    minManhattan = 999999
    for pos in foodPositions:
        dist = util.manhattanDistance( position, pos )
        if dist < minManhattan:
            minManhattan = dist
    allManhattan = (allManhattan + minManhattan)/2
    return allManhattan

# nearest neighbor plus number of food left
def NNManhattanHeuristic(state, problem):
    position, foodGrid = state
    foodPositions = foodGrid.asList()
    if len(foodPositions)==0:return 0
    minManhattan = 999999
    for pos in foodPositions:
        dist = util.manhattanDistance( position, pos )
        if dist < minManhattan:
            minManhattan = dist
    estimate = minManhattan + len(foodPositions) - 1
    return estimate

###########################################################################
# construct a minimum spanning tree; use the weight as heuristic value
###########################################################################

def MSTEuclideanHeuristic(state, problem):
    
    position, foodGrid = state
    foodPositions = foodGrid.asList()
    if len(foodPositions)==0:return 0
    
    totalLength = 0
    MST = [position]
    while len(foodPositions) != 0:
        xy1,xy2,minimum = findClosestEuclideanPair(MST,foodPositions)

        #xy1,xy2,minimum = findClosestEuclideanPair2(MST,foodPositions,state)
        #print xy1,xy2,minimum
        totalLength += minimum
        MST.append(xy2)
        foodPositions.remove(xy2)
    return totalLength

# find the cloest pair of points in two sets
# return a triple in the form of (point1, point2, distance)
def findClosestEuclideanPair(positions1,positions2):
    minimum = float('inf')
    position1 = None
    position2 = None
    for pos1 in positions1:
        for pos2 in positions2:
            #distance = util.manhattanDistance (pos1,pos2)
            distance = ((pos1[0] - pos2[0])**2+(pos1[1] - pos2[1])**2)**0.5
            if distance < minimum:
                minimum = distance
                position1 = pos1
                position2 = pos2
    return (position1,position2,minimum)

###########################################################################
# construct a rectilinear steiner minimum spanning tree
###########################################################################

def RectilinearMinimumSteinerSpanningTreeHeuristic(state,problem):
    position, foodGrid = state
    foodPositions = foodGrid.asList()
    if len(foodPositions)==0:return 0

    P = copy.deepcopy(foodPositions) + [position]
    S=[]
    Candidate_Set = computeCandidateSet(P,S)
    while len(Candidate_Set) != 0:
        # Step1: find x in Candidate Set which maximizes delta_MST (P union S, {x})
        maximumDelta = -float('inf')
        x = None
        for candidate in Candidate_Set:
            #print maximumDelta
            #print candidate
            delta = computeDeltaMST(P+S,[candidate])
            #print delta
            if delta > maximumDelta:
                maximumDelta = delta
                x = candidate
        # Step2: update S
        S = S+[x]

        # Step3: Remove points in S which have degree <= 2 in MST(P union S) 
        remove_foo(P,S)
        #P=[(1, 2), (6, 2), (4, 3)]
        '''print "*********"
        print Candidate_Set
        print P
        print S'''
        Candidate_Set = computeCandidateSet(P,S) 
    return computeMSTCost(P+S)

#A is a list of points; A is not []
def computeMSTCost(lst):
    A = copy.deepcopy(lst)
    totalLength = 0
    MST = [A.pop()]
    while len(A) != 0:
        xy1,xy2,minimum = findClosestManhattanPair(MST,A)
        totalLength += minimum
        MST.append(xy2)
        A.remove(xy2)
    return totalLength

def remove_foo(P,S):
    A = P+S
    MST = [A.pop()]
    MSTWithDegree = {MST[0]:0}
    while len(A) != 0:
        xy1,xy2,minimum = findClosestManhattanPair(MST,A)
        MST.append(xy2)
        MSTWithDegree[xy1] += 1
        MSTWithDegree[xy2] =1
        A.remove(xy2)

    for key in MSTWithDegree:
        if key in S and MSTWithDegree[key] <3:
            S.remove(key)

def computeCandidateSet(P,S): 
    H = computeH(P+S)
    cpy_H = copy.deepcopy(H)
    for candidate in cpy_H:
        delta = computeDeltaMST(P+S,[candidate])
        if delta <= 0:
            H.remove(candidate)
    return H

#compute H(P),  i.e., the intersection points of all horizontal and
#vertical lines passing through points of P
def computeH(foo):
    H = []
    # Step 1: find all possible x and y
    xSet=[]
    ySet=[]
    for xy in foo:
        x = xy[0]
        y = xy[1]
        if x not in xSet:
            xSet.append(x)
        if y not in ySet:
            ySet.append(y)

    # Step 2: find all steiner candidates
    for x in xSet:
        for y in ySet:
            if (x,y) not in foo:
                H.append((x,y))

    return H

#delta_MST(A, B) = cost(MST(A)) -cost(MST(A union B))
def computeDeltaMST(A,B):
    return computeMSTCost(A) - computeMSTCost(A+B)

# find the cloest pair of points in two sets
# return a triple in the form of (point1, point2, distance)
# this helpful function is also used by MSTManhattanHeuristicWithWallPenaltyHeuristic
def findClosestManhattanPair(positions1,positions2):
    minimum = float('inf')
    position1 = None
    position2 = None
    for pos1 in positions1:
        for pos2 in positions2:
            #distance = util.manhattanDistance (pos1,pos2)
            distance = abs(pos1[0] - pos2[0])+abs(pos1[1] - pos2[1])
            if distance < minimum:
                minimum = distance
                position1 = pos1
                position2 = pos2
    return (position1,position2,minimum)

###########################################################################
# an optimization of MSTEuclideanHeuristic
# use Manhataan distance insteaed of Euclidena distance
# use wall penalty (core idea)
###########################################################################


def MSTManhattanHeuristicWithWallPenaltyHeuristic(state, problem):

    wallsAlreadyPenalized = []

    walls = copy.deepcopy(problem.walls)

    position, foodGrid = state
    foodPositions = foodGrid.asList()

    # goal test
    if len(foodPositions)==0:return 0

    cpy = copy.deepcopy(foodPositions)
    totalLength = 0
    MST = [cpy.pop()]

    while len(cpy) != 0:

        xy1,xy2,minimum = findClosestManhattanPair(MST,cpy)

        totalLength += minimum
        MST.append(xy2)
        cpy.remove(xy2)

        # give horizontal wall penalty here
        # not all walls information can be used

        #totalLength += wallPenaltyFunction(xy1,xy2,walls)
        wallsPenalized = wallPenaltyOnlyOnceFunction(xy1,xy2,walls)
        wallsPenalized = list(set(wallsPenalized)-set(wallsAlreadyPenalized))
        wallsAlreadyPenalized += wallsPenalized
        totalLength += len(wallsPenalized) *2
        #print wallsAlreadyPenalized

        '''IN PrOGRESS'''
        #if abs(xy1[0]-xy2[0]) == 2 and abs(xy1[1]-xy2[1]) == 2:

        '''ENF OF IN PrOGRESS'''


    # an optimization here:
    #   do not include the pacaman location in the MST first; add it at the end
    xy1,xy2,minimum = findClosestManhattanPair([position],MST)
    totalLength += minimum

    # very important here; still need to consider penalty (or the heuristic is not consistent)
    #totalLength += wallPenaltyFunction(xy1,xy2,walls)
    wallsPenalized = wallPenaltyOnlyOnceFunction(xy1,xy2,walls)
    wallsPenalized = list(set(wallsPenalized)-set(wallsAlreadyPenalized))
    wallsAlreadyPenalized += wallsPenalized
    totalLength += len(wallsPenalized) *2
    #print "****"
    return totalLength

'''Currently, to keep admissibility, only horizontal penalty is considered
Return a list of penalized walls
'''
def wallPenaltyOnlyOnceFunction(xy1,xy2,walls):

    y1 = min(xy1[1],xy2[1])
    y2 = max(xy1[1],xy2[1])
    x1 = min(xy1[0],xy2[0])
    x2 = max(xy1[0],xy2[0])


    '''IN PROGRESS'''
    # to make sure it is admissible
    if y2 - y1 > 3:
        #return penalty
        pass
    '''END OF IN PROGRESS'''

    max_r_p = 0
    r_p_walls = []
    max_l_p = 0
    l_p_walls = []
    for y in range(y1+1,y2):
        lst = map(lambda x: walls[x][y], range(x1,x2+1))
        if False not in lst:
            r_p,l_p = (horizontalBidirectionWallSearch(x1,x2,y,walls))
            if r_p > max_r_p:
                max_r_p = r_p
                if r_p != float('inf'):
                    r_p_walls = map(lambda i: (x2+i,y), range(r_p))
            if l_p > max_l_p:
                max_l_p = l_p
                if l_p != float('inf'):
                    l_p_walls = map(lambda i: (x2-i,y), range(l_p))

    if max_r_p < max_l_p:
        return r_p_walls
    else:
        return l_p_walls


def wallPenaltyFunction(xy1,xy2,walls):

    penalty = 0

    y1 = min(xy1[1],xy2[1])
    y2 = max(xy1[1],xy2[1])
    x1 = min(xy1[0],xy2[0])
    x2 = max(xy1[0],xy2[0])


    '''IN PROGRESS'''
    # to make sure it is admissible
    if y2 - y1 > 3:
        #return penalty
        pass
    '''END OF IN PROGRESS'''

    # 1. horizontal walls
    max_r_p = 0
    max_l_p = 0
    for y in range(y1+1,y2):
        lst = map(lambda x: walls[x][y], range(x1,x2+1))
        if False not in lst:
            r_p,l_p = (horizontalBidirectionWallSearch(x1,x2,y,walls))
            if r_p > max_r_p:
                max_r_p = r_p
            if l_p > max_l_p:
                max_l_p = l_p


    # 2. vertical walls
    max_u_p = 0 #up
    max_d_p = 0 #down
    for x in range(x1+1,x2):
        lst = map(lambda y: walls[x][y], range(y1,y2+1))
        if False not in lst:
            u_p,d_p = (verticalBidirectionWallSearch(y1,y2,x,walls))
            if u_p > max_u_p:
                max_u_p = u_p
            if d_p > max_d_p:
                max_d_p = d_p

    # two cases depends on the slope
    '''IN PROGRESS'''
    max_u_p = 0 #up
    max_d_p = 0 #down
    '''END OF IN PROGRESS'''
    slope = None
    if x1==x2:
        slope = float('inf')
    else:
        slope = (xy2[1] - xy1[1])*1.0 / (xy2[0] - xy1[0])
    if slope > 0:
        penalty = min(max_r_p+max_d_p,max_l_p+max_u_p)
    else:
        penalty = min(max_r_p+max_u_p,max_l_p+max_d_p)


    return penalty
    
# x1, x2 is the right, left search begining position
# y is the y-position of the search
def horizontalBidirectionWallSearch(x1,x2,y,walls):
    r_penalty = 0
    x_of_right_break_seeker = x2  

    while walls[x_of_right_break_seeker][y]:
        r_penalty += 2
        x_of_right_break_seeker +=1
        # handle boundry issue
        if x_of_right_break_seeker == walls.width:
            r_penalty = float('inf')
            break

    l_penalty = 0
    x_of_left_break_seeker = x1
    while walls[x_of_left_break_seeker][y]:
        l_penalty += 2
        x_of_left_break_seeker -=1
        # handle boundry issue
        if x_of_left_break_seeker == 0:
            l_penalty = float('inf')
            break

    return (r_penalty,l_penalty) 

# x1, x2 is the right, left search begining position
# y is the y-position of the search
def verticalBidirectionWallSearch(y1,y2,x,walls):
    u_penalty = 0
    y_of_up_break_seeker = y2  

    while walls[x][y_of_up_break_seeker]:
        u_penalty += 2
        y_of_up_break_seeker +=1
        # handle boundry issue
        if y_of_up_break_seeker == walls.height:
            u_penalty = float('inf')
            break

    d_penalty = 0
    y_of_down_break_seeker = y1
    while walls[x][y_of_down_break_seeker]:
        d_penalty += 2
        y_of_down_break_seeker -=1
        # handle boundry issue
        if y_of_down_break_seeker == 0:
            d_penalty = float('inf')
            break

    return (u_penalty,d_penalty) 

###########################################################################
# END OF QUESTION # 7
###########################################################################


class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        return search.breadthFirstSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        "*** YOUR CODE HERE ***"
        return state in self.food.asList()
        

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
