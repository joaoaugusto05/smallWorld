import random
import copy
from network import graph
from search import dfs, AStar, bestFirst, bfs, dijkstra, experiments
import sys

def main():
    sys.setrecursionlimit(5000) 
    # Example graph
    n = 100
    g = graph(n)

    for i in range(n):
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        g.addNode(i, x, y)

    g.setSmallWorld(7, 0.01)

    # Examples
    dfs(g, 1, 2, verbose=True)
    g.plot(savingPath='../output/img/dfs_example.png',labels=False)
    g.reset()
    AStar(g, 1, 2, verbose=True)
    g.plot(savingPath='../output/img/AStar_example.png',labels=False)
    g.reset()
    bestFirst(g, 1, 2, verbose=True)
    g.plot(savingPath='../output/img/bestFirst_example.png',labels=False)
    g.reset()
    bfs(g, 1, 2, verbose=True)
    g.plot(savingPath='../output/img/bfs_example.png',labels=False)
    g.reset()
    dijkstra(g, 1, 2, verbose=True)
    g.plot(savingPath='../output/img/dijkstra_example.png',labels=False)
    g.reset()

    n = 2000
    g = graph(n)

    for i in range(n):
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        g.addNode(i, x, y)

    g.setSmallWorld(7, 0.01)
    results = experiments(g, ['dfs', 'AStar', 'bestFirst', 'bfs', 'dijkstra'], savingPath='../output/experiments/results_n2000_k7_p1.json')

    n = 2000
    g = graph(n)

    for i in range(n):
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        g.addNode(i, x, y)

    g.setSmallWorld(7, 0.05)
    results = experiments(g, ['dfs', 'AStar', 'bestFirst', 'bfs', 'dijkstra'], savingPath='../output/experiments/results_n2000_k7_p5.json')

    n = 2000
    g = graph(n)

    for i in range(n):
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        g.addNode(i, x, y)

    g.setSmallWorld(7, 0.1)
    results = experiments(g, ['dfs', 'AStar', 'bestFirst', 'bfs', 'dijkstra'], savingPath='../output/experiments/results_n2000_k7_p10.json')
    
if __name__ == "__main__":
    main()