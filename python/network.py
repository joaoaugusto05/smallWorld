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
        self.adjacencyList = [{} for _ in range(n)]
        self.nodes = [None] * n
        self.pathed = False

    def addEdge(self, src, dest):
        if not dest in self.adjacencyList[src]:
            self.adjacencyList[src][dest] = np.linalg.norm(np.array([self.nodes[src].x, self.nodes[src].y]) - np.array([self.nodes[dest].x, self.nodes[dest].y]))
        if not src in self.adjacencyList[dest]:
            self.adjacencyList[dest][src] = np.linalg.norm(np.array([self.nodes[src].x, self.nodes[src].y]) - np.array([self.nodes[dest].x, self.nodes[dest].y]))

    def getNeighbors(self, nodeID):
        return self.adjacencyList[nodeID].keys()

    def addNode(self, i, x, y):
        self.nodes[i] = node(i, x, y)

    def closestNodes(self, nodeIndex, nodeList, n):
        nodeA = nodeList[nodeIndex]
        coords = np.array([[node.x, node.y] for node in nodeList])
        distances = np.linalg.norm(coords - [nodeA.x, nodeA.y], axis=1)
        distances[nodeIndex] = np.inf
        closest_indices = np.argsort(distances)[:n]
        [self.addEdge(nodeIndex, closest_indices[j]) for j in range(n)]
        
    def setRandomize(self, p, closeList, idx):
        keys = list(closeList.keys()).copy()
        for i in keys:
            chance = random.random()
            if chance < p:
                newVal = random.randint(0, self.n - 1)
                while newVal == i | newVal in keys:
                    newVal = random.randint(0, self.n - 1)
                closeList.pop(i)
                self.adjacencyList[i].pop(idx)
                self.addEdge(i,newVal)

    def setSmallWorld(self, n, p):
        [self.closestNodes(i, self.nodes, n) for i in range(self.n)] 
        [self.setRandomize(p, closeList, idx) for idx,closeList in enumerate(self.adjacencyList)]

    def printSmallWorld(self):
        for i in range(len(self.nodes)):
            print("-------------------------------------------------")
            print(self.nodes[i].i, self.nodes[i].x, self.nodes[i].y)
            print("Proximo aos nós: ")
            print(self.adjacencyList[i].keys())
            print("-------------------------------------------------")
    
    def reset(self):
        self.path = []
        self.pathed = False

    def plot(self, savingPath=None, labels=False, size=(15,10)):
        fig, ax = plt.subplots(figsize=size)

        X_pos = np.array([[nodeA.x] for nodeA in self.nodes])
        Y_pos = np.array([[nodeA.y] for nodeA in self.nodes])
        ax.scatter(X_pos, Y_pos, color = 'blue')

        if self.pathed:
            Xinicio = self.nodes[self.path[0]].x
            Yinicio = self.nodes[self.path[0]].y
            ax.scatter(Xinicio, Yinicio, color = 'green', s = 100)

            Xfim = self.nodes[self.path[-1]].x
            Yfim = self.nodes[self.path[-1]].y
            ax.scatter(Xfim, Yfim, color = 'red', s = 100)

            XposPath = []
            YposPath = []
            for i in self.path[1:-1]:
                XposPath.append(self.nodes[i].x)
                YposPath.append(self.nodes[i].y)
            ax.scatter(XposPath, YposPath, color = 'magenta',s = 75)

            if labels:
                for idx, node in enumerate(self.nodes):
                    ax.annotate(idx,(node.x,node.y+0.20))
            
        for i in range(len(self.nodes)):
            for j in self.adjacencyList[i].keys():
                if self.nodes[i].i in self.path and self.nodes[j].i in self.path:
                    ax.plot([self.nodes[i].x, self.nodes[j].x], [self.nodes[i].y, self.nodes[j].y], c= 'darkorange', linewidth = 2)
                else:
                    ax.plot([self.nodes[i].x, self.nodes[j].x], [self.nodes[i].y, self.nodes[j].y], c= 'black', linewidth = 0.5)
                if labels:
                    ax.annotate(f"{self.adjacencyList[i][j]:.2f}",((self.nodes[i].x+self.nodes[j].x)/2,(self.nodes[i].y+self.nodes[j].y)/2))

        if savingPath:
            plt.savefig(savingPath)
        plt.show()
