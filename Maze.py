import heapq
import random
import matplotlib.pyplot as plt
import numpy as np


def generate_maze(width, height):
    # DFS needs a odd size to generate matrix, also ensure that 2 node move does not go out of bounds
    if width % 2 == 0: width += 1
    if height % 2 == 0: height += 1

    maze = np.ones((height, width), dtype=int)
    stack = [(1, 1)]
    maze[1, 1] = 0

    while stack:
        x, y = stack[-1] #latest stack node
        neighbors = []
        for dx, dy in [(-2,0), (2,0), (0,-2), (0,2)]:
            nx, ny = x+dx, y+dy
            if 0 < nx < height-1 and 0 < ny < width-1 and maze[nx, ny]:
                neighbors.append((nx, ny))

        if neighbors:
            nx, ny = random.choice(neighbors)
            maze[(x+nx)//2, (y+ny)//2] = 0 #set path to neighbor open
            maze[nx, ny] = 0 #set neighbor open
            stack.append((nx, ny))
        else:
            stack.pop()

    maze[height-2, width-2] = 0 #set exit

    return maze, (1,1), (height-2, width-2)

def heuristic(node, end):
    return abs(node[0] - end[0]) + abs(node[1] - end[1])

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1] #reverse the list to show the correct path.

def dynamic_astar_visualization(maze, start, end):
    open_list = []
    closed_list = set()
    came_from = {}
    start_node = (start[0], start[1])
    end_node = (end[0], end[1])

    heapq.heappush(open_list, (0 + heuristic(start, end), 0, start_node)) #push f, g, start node to open_list

    plt.figure(figsize=(10, 10))

    path = None
    while open_list:
        _, g, current = heapq.heappop(open_list)
        x, y = current

        closed_list.add(current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: #4 side search
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx, ny] == 0: #inside the boundary check
                neighbor = (nx, ny)
                if neighbor not in closed_list: # not visited check
                    tentative_g = g + 1
                    heapq.heappush(open_list, (tentative_g + heuristic(neighbor, end_node), tentative_g, neighbor))
                    came_from[neighbor] = current

        plt.clf()
        open_nodes = [node for _, _, node in open_list]
        for node in open_nodes:
            plt.scatter(node[1], node[0], color="pink", s=5)

        plt.imshow(maze, cmap="Blues", origin="upper")

        for node in closed_list:
            plt.scatter(node[1], node[0], color="red", s=5) #set closed node red

        if current != start_node:
            plt.scatter(y, x, color="green", s=5) #set current node green

        plt.title(f"A* Algorithm: Current Node: {current}")
        plt.draw()
        plt.pause(0.05)

        if current == end_node:
            path = reconstruct_path(came_from, current)
            plt.clf() #clear image to show a new image.
            plt.imshow(maze, cmap="Blues", origin="upper")
            open_nodes = [node for _, _, node in open_list]
            for node in open_nodes:
                plt.scatter(node[1], node[0], color="pink", s=5)
            for node in closed_list:
                plt.scatter(node[1], node[0], color="red", s=5)
            if path:
                for p in path:
                    plt.scatter(p[1], p[0], color="purple", s=5)
            plt.scatter(start[1], start[0], color="green", s=5)
            plt.scatter(end[1], end[0], color="blue", s=5)
            plt.title("A* Algorithm: Path Found!")
            plt.draw()
            plt.pause(0.05)
            break

    if not path:
        print("No path found!")
    plt.show()


maze, start, end = generate_maze(21, 21)
dynamic_astar_visualization(maze, start, end)
