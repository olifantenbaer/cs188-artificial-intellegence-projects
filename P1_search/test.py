class Node:
    def __init__(self,cost):
        #self.parent = parent
        #self.length = length
        #self.action = action # the action which taken at parent state
        self.cost = cost
        #self.state = state


import util

n1 = Node(4)

n2= Node(5)

n3 = Node(4)

f = util.Stack()

f.push(n1)
f.push(n2)

n1=7
n1 = f.pop()



print n1.cost

