import pickle

# This file is for simplifying the search of the dynamics function
# Nodes are states. Edges are actions between states

action_space = 9
reset_stats = False

class Node:
    def __init__(self):
        self.visits = 0
        self.edges = {}
        for x in range(action_space):
            self.edges[x] = Edge()

class Edge:
    def __init__(self):
        self.visits = 0
        self.value_sum = 0
        self.last_destination = None

class Graph:
    def __init__(self):
        self.action_space = action_space
        try:
            self.nodes = pickle.load(open('graph.p','rb'))
        except:
            self.nodes = {}
    
    def save_graph(self):
        pickle.dump(self.nodes, open('graph.p','wb'))

    def traverse(self, node, action, value, dest):
        if not (node in self.nodes):
            # Node never visited before, must create new node with new statistics
            n = self.nodes[node] = Node()
        else:
            n = self.nodes[node]
        e = n.edges[action]
        n.visits += 1
        e.visits += 1
        e.value_sum += value
        if e.last_destination != dest:
            e.last_destination = dest
            # Resets statistics if the destination node changes
            if reset_stats:
                e.visits = 1
                e.value_sum = value

    # For filling in assumed edge values by prediction of the dynamics function
    def imagine(self, node, action, value, dest):
        if node in self.nodes:
            edge = self.nodes[node].edges[action]
            edge.value_sum = value
            edge.visits = 1
            edge.last_destination = dest

    def search(self, node):
        if node in self.nodes:
            return self.nodes[node]
        else:
            return None

# Testing
if __name__ == '__main__':
    g = Graph()
    g.traverse('00', 0, 1, '01')
    g.save_graph()
    g = Graph()
    node = g.search('00')
    if node == None:
        print('Pickle did not load correctly')
    else:
        print('Success', node.edges[0].value_sum)