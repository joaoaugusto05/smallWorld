import numpy as np
import random
from queue import PriorityQueue
from collections import deque
import time
import json

def dfsUtil(graph, node, goal, visited, parent, pathCost):
    visited[node] = True
    if node == goal:
        return True
    for neighbor in graph.getNeighbors(node):
        if not visited[neighbor]:
            parent[neighbor] = node
            pathCost[neighbor] = pathCost[node] + graph.adjacencyList[node][neighbor]
            if dfsUtil(graph, neighbor, goal, visited, parent, pathCost):
                return True
    return False

def dfs(graph, nodeO, nodeD, verbose=False):
    graph.pathed = True
    graph.path = []
    visited = np.zeros(graph.n, dtype=bool)
    parent = np.full(graph.n, -1)
    pathCost = np.zeros(graph.n)
    found = dfsUtil(graph, nodeO, nodeD, visited, parent, pathCost)
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
        while curr != -1:
            graph.path.insert(0, curr)
            curr = parent[curr]
        if verbose:
            print(f"Path: {' -> '.join(map(str,graph.path))}")
            print(f"Cost: {pathCost[nodeD]:.3f}")
        return graph.path, pathCost[nodeD]

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
            graph.path.append(nodeO)
            graph.path.append(nodeD)
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

def heuristic(nodeA, nodeB):
    return np.linalg.norm(np.array([nodeA.x, nodeA.y]) - np.array([nodeB.x, nodeB.y]))

def bestFirst(graph, nodeO, nodeD, verbose=False):
    frontier = PriorityQueue()
    frontier.put((0, nodeO))
    came_from = {nodeO: None}
    currentCost = {nodeO: 0}

    while not frontier.empty():
        _, current = frontier.get()
        if current == nodeD:
            break
        for next_node in graph.getNeighbors(current):
            if next_node not in came_from:
                cost = currentCost[current] + graph.adjacencyList[current][next_node]
                currentCost[next_node] = cost
                priority = heuristic(graph.nodes[nodeD], graph.nodes[next_node])
                frontier.put((priority, next_node))
                came_from[next_node] = current
    
    graph.pathed = True
    graph.path = []
    if verbose:
        print("\nBest-First")
    if nodeD not in came_from:
        graph.path.append(nodeO)
        graph.path.append(nodeD)
        if verbose:
            print("No path was found!")
        return -1
    else:
        current = nodeD
        while current is not None:
            graph.path.append(current)
            current = came_from[current]
        graph.path.reverse()
        if verbose:
                print(f"Path: {' -> '.join(map(str,graph.path))}")
                print(f"Cost: {currentCost[nodeD]:.3f}")
        return graph.path, currentCost[nodeD]

def bfs(graph, nodeO, nodeD, verbose=False):
    queue = deque()

    visited = [False] * (graph.n)
  
    parent = [-1] * (graph.n)

    queue.append(nodeO)
    visited[nodeO] = True

    graph.pathed = True
    graph.path = []

    if verbose:
        print("\nBFS")

    while queue:
        currentNode = queue.popleft()
  
        if currentNode == nodeD:
            while currentNode != -1:
                graph.path.append(currentNode)
                currentNode = parent[currentNode]
            graph.path = graph.path[::-1]
            cost = sum(graph.adjacencyList[graph.path[i]][graph.path[i+1]] for i in range(len(graph.path)-1))
            if verbose:
                if verbose:
                    print(f"Path: {' -> '.join(map(str,graph.path))}")
                    print(f"Cost: {cost:.3f}")
            return graph.path, cost
  
        for i in graph.getNeighbors(currentNode):
            if visited[i] == False:
                queue.append(i)
                visited[i] = True
                parent[i] = currentNode
    graph.path.append(nodeO)
    graph.path.append(nodeD)
    if verbose:
        print("No path found!")
    return -1

def dijkstra(graph, nodeO, nodeD, verbose=False):
    queue = PriorityQueue()
    queue.put((0, nodeO))
    visited = set()
    predecessors = {nodeO: None}
    distances = {nodeO: 0}

    graph.pathed = True
    graph.path = []

    if verbose:
        print("\nDijkstra")
    while not queue.empty():
        (dist, currentNode) = queue.get()

        visited.add(currentNode)

        if currentNode == nodeD:
            total_cost = distances[currentNode]
            while currentNode is not None:
                graph.path.append(currentNode)
                currentNode = predecessors[currentNode]
            graph.path.reverse()
            if verbose:
                print(f"Path: {' -> '.join(map(str,graph.path))}")
                print(f"Cost: {total_cost:.3f}")
            return graph.path, total_cost

        for neighbor in graph.getNeighbors(currentNode):
            new_dist = distances[currentNode] + graph.adjacencyList[currentNode][neighbor]

            if neighbor not in visited and (neighbor not in distances or new_dist < distances[neighbor]):
                distances[neighbor] = new_dist
                predecessors[neighbor] = currentNode
                queue.put((new_dist, neighbor))
    graph.path.append(nodeO)
    graph.path.append(nodeD)
    if verbose:
        print("No path was found!")
    raise -1

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
                if search == -1:
                    results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':-1, 'cost':-1, 'time':(endTime - startTime)}
                elif len(search) > 1:
                    results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':search[0], 'cost':search[1], 'time':(endTime - startTime)}
                    
            elif method == 'AStar':
                startTime = time.time()
                search = AStar(graph,nodes[i],nodes[i+1])
                endTime = time.time()
                if search == -1:
                    results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':-1, 'cost':-1, 'time':(endTime - startTime)}
                elif len(search) > 1:
                    results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':search[0], 'cost':search[1], 'time':(endTime - startTime)}

            elif method == 'bestFirst':
                startTime = time.time()
                search = bestFirst(graph,nodes[i],nodes[i+1])
                endTime = time.time()
                if search == -1:
                    results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':-1, 'cost':-1, 'time':(endTime - startTime)}
                elif len(search) > 1:
                    results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':search[0], 'cost':search[1], 'time':(endTime - startTime)}

            elif method == 'bfs':
                startTime = time.time()
                search = bfs(graph,nodes[i],nodes[i+1])
                endTime = time.time()
                if search == -1:
                    results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':-1, 'cost':-1, 'time':(endTime - startTime)}
                elif len(search) > 1:
                    results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':search[0], 'cost':search[1], 'time':(endTime - startTime)}

            elif method == 'dijkstra':
                startTime = time.time()
                search = dijkstra(graph,nodes[i],nodes[i+1])
                endTime = time.time()
                if search == -1:
                    results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':-1, 'cost':-1, 'time':(endTime - startTime)}
                elif len(search) > 1:
                    results[method][i//2] = {'nodeO':nodes[i], 'nodeD':nodes[i+1], 'path':search[0], 'cost':search[1], 'time':(endTime - startTime)}
                
    if savingPath:
        with open(savingPath, "w") as outfile:
            json.dump(results, outfile, cls=NpEncoder)
    return results