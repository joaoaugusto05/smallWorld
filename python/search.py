import numpy as np
import random
from queue import PriorityQueue
import time
import json

def dfsUtil(graph, curr, destination, visited, parent):
    visited[curr] = True
    if curr == destination:
        return True
    for vizinho in graph.adjacencyList[curr].keys():
        if not visited[vizinho]:
            parent[vizinho] = curr
            if dfsUtil(graph, vizinho, destination, visited, parent):
                return True
    return False
    
def dfs(graph, nodeO, nodeD, verbose=False):
    graph.pathed = True
    graph.path = []
    visited = np.zeros(graph.n, dtype=bool)
    parent = np.full(graph.n, -1)
    found = dfsUtil(graph, nodeO, nodeD, visited, parent)
    if verbose:
        print("\nDFS")
    if not found:
        graph.path.append(nodeO)
        graph.path.append(nodeD)
        if verbose:
            print("No path was found!")
            return -1
    else:
        curr = nodeD
        #graph.path.append(curr)
        while curr != -1:
            graph.path.insert(0, curr)
            curr = parent[curr]
        if verbose:
            print(f"Path: {' -> '.join(map(str,graph.path))}")
        return graph.path

def AStar(graph, nodeO, nodeD, verbose=False):
        frontier = PriorityQueue()
        frontier.put((0, nodeO))
        
        cameFrom = {nodeO: None}
        currentCost = {nodeO: 0}
        
        while not frontier.empty():
            current = frontier.get()[1]         
            if current == nodeD:
                break         
            for next_node in graph.getNeighbors(current):
                newCost = currentCost[current] + graph.adjacencyList[current][next_node]
                if next_node not in currentCost or newCost < currentCost[next_node]:
                    currentCost[next_node] = newCost
                    priority = newCost + np.linalg.norm(np.array([graph.nodes[nodeD].x, graph.nodes[nodeD].y]) - np.array([graph.nodes[next_node].x, graph.nodes[next_node].y]))
                    frontier.put((priority, next_node))
                    cameFrom[next_node] = current
        graph.pathed = True
        graph.path = []
        if verbose:
            print("\nA*")
        if nodeD not in cameFrom:
            if verbose:
                print("No path was found!")
            return -1
        else:
            current = nodeD
            while current is not None:
                graph.path.append(current)
                current = cameFrom[current]
            graph.path.reverse()
            if verbose:
                print(f"Path: {' -> '.join(map(str,graph.path))}")
                print(f"Cost: {currentCost[nodeD]:.3f}")
            return graph.path, currentCost[nodeD]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def experiments(graph, searchMethods, nPairs=10, savingPath=None):
    results = {}
    for i in searchMethods:
        results[i] = {}
    if nPairs*2 > len(graph.nodes):
        print("Not enough nodes")
        return {}
    nodes = random.sample(range(len(graph.nodes)),2*nPairs)
    for i in range(0,len(nodes),2):
        for method in searchMethods:
            if method == 'dfs':
                startTime = time.time()
                search = dfs(graph,nodes[i],nodes[i+1])
                endTime = time.time()
                results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':search, 'time':(endTime - startTime)}
                    
            elif method == 'AStar':
                startTime = time.time()
                search = AStar(graph,nodes[i],nodes[i+1])
                endTime = time.time()
                if search == -1:
                    results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':-1, 'cost':-1, 'time':(endTime - startTime)}
                elif len(search) > 1:
                    results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':search[0], 'cost':search[1], 'time':(endTime - startTime)}
                
    if savingPath:
        with open(savingPath, "w") as outfile:
            json.dump(results, outfile, cls=NpEncoder)
    return results