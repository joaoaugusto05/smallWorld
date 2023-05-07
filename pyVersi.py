import numpy as np
import random
import matplotlib.pyplot as plt

class node:
    def __init__(self, i, x, y):
        self.i = i
        self.x = x
        self.y = y
        

class graph:
    def __init__(self, n):
        self.n = n
        self.adjacencyList = [[] for _ in range(n)]
        self.nodes = [None] * n
        self.pathed = False

    def addEdge(self, src, dest):
        if not dest in self.adjacencyList[src]:
            self.adjacencyList[src].append(dest)
        if not src in self.adjacencyList[dest]:
            self.adjacencyList[dest].append(src)
    def addNode(self, i, x, y):
        self.nodes[i] = node(i, x, y)

    def closestNodes(self, nodeIndex, nodeList, n):
        nodeA = nodeList[nodeIndex]
        coords = np.array([[node.x, node.y] for node in nodeList])
        distances = np.linalg.norm(coords - [nodeA.x, nodeA.y], axis=1)
        distances[nodeIndex] = np.inf
        closest_indices = np.argsort(distances)[:n]
        [self.addEdge(nodeIndex, closest_indices[j]) for j in range(n)]
        
    def setRandomize(self, p, closeList):
        for i in range(len(closeList)):
            chance = random.random()
            if chance < p:
                newVal = random.randint(0, self.n - 1)
                while newVal == i | newVal in closeList:
                    newVal = random.randint(0, self.n - 1)
                closeList[i] = newVal

    def setSmallWorld(self, n, p):
        [self.closestNodes(i, self.nodes, n) for i in range(self.n)] 
        [self.setRandomize(p, closeList) for closeList in self.adjacencyList] 
    def printSmallWorld(self):
        for i in range(len(self.nodes)):
            print("-------------------------------------------------")
            print(self.nodes[i].i, self.nodes[i].x, self.nodes[i].y)
            print("Proximo aos nós: ")
            print(self.adjacencyList[i])
 
            print("-------------------------------------------------")
    
    def plot(self):
        #Plotar nós
        X_pos = np.array([[nodeA.x] for nodeA in self.nodes])
        Y_pos = np.array([[nodeA.y] for nodeA in self.nodes])
        plt.scatter(X_pos, Y_pos, color = 'blue')
        if self.pathed:
            Xinicio = self.nodes[self.path[0]].x
            Yinicio = self.nodes[self.path[0]].y
            plt.scatter(Xinicio, Yinicio, color = 'green', s = 100)

            Xfim = self.nodes[self.path[-1]].x
            Yfim = self.nodes[self.path[-1]].y
            plt.scatter(Xfim, Yfim, color = 'red', s = 100)

            XposPath = []
            YposPath = []
            for i in self.path:
                XposPath.append(self.nodes[i].x)
                YposPath.append(self.nodes[i].y)
            plt.scatter(XposPath, YposPath, color = 'black')
        #Plotar Arestas
        X = []
        Y = []
        for i in range(len(self.nodes)):
            for j in self.adjacencyList[i]:
                plt.plot([self.nodes[i].x, self.nodes[j]. x], [self.nodes[i].y, self.nodes[j].y], c= 'black', linewidth = 0.5)
        plt.show()

    def dfsUtil(self, curr, destination, visited, parent):
        visited[curr] = True
        if curr == destination:
            return True
        for vizinho in self.adjacencyList[curr]:
            if not visited[vizinho]:
                parent[vizinho] = curr
                if self.dfsUtil(vizinho, destination, visited, parent):
                    return True
        return False

    def dfs(self, nodeO, nodeD):

        self.pathed = True
        self.path = []

        visited = np.zeros(self.n, dtype=bool)
        parent = np.full(self.n, -1)
        found = self.dfsUtil(nodeO, nodeD, visited, parent)
        if not found:
            self.path.append(nodeO)
            self.path.append(nodeD)
            print("Nao há caminho")
        else:
            
            
            curr = nodeD
            self.path.append(curr)
            while curr != -1:
                self.path.insert(0, curr)
                curr = parent[curr]
            print(self.path)


n = 10
g = graph(n)

for i in range(n):
    x = random.randint(0, n-1)
    y = random.randint(0, n-1)
    g.addNode(i, x, y)

g.setSmallWorld(1, 0.1)
#g.printSmallWorld()
g.dfs(1, 2)
g.plot()
