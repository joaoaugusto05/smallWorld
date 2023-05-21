import random

from network import graph
from search import dfs, AStar, experiments

def main():
    n = 100
    g = graph(n)

    for i in range(n):
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        g.addNode(i, x, y)

    g.setSmallWorld(3, 0.01)
    #g.printSmallWorld()
    dfs(g, 1, 2, verbose=True)
    AStar(g, 1, 2, verbose=True)

    results = experiments(g, ['dfs','AStar'], savingPath='../output/experiments/results.json')

    g.plot(savingPath='../output/img/network.png',labels=False)

    
if __name__ == "__main__":
    main()