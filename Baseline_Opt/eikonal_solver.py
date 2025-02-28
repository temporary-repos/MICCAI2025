import numpy as np
import heapq

class EikonalSolverPointCloud:
    def __init__(self, points, faces, velocity_field):
        self.points = points
        self.faces = faces
        self.velocity_field = velocity_field
    
    def get_vertex_neighbors(self, vertex_index):
        # Find neighboring vertices for a given node based on the mesh faces
        neighbors = set()
        for fi in np.where(self.faces == vertex_index)[0]:
            face = self.faces[fi]
            for vi in face:
                if vi != vertex_index:
                    neighbors.add(vi)
        return neighbors

    def solve(self, source_indices):
        # Initialize travel times with a large value
        travel_times = np.full(len(self.points), np.inf)
        
        # Set the travel time of source points to 0
        travel_times[source_indices] = 0
        
        # Use a min heap as the 'front' of the wave propagation
        front = [(0, src) for src in source_indices]
        heapq.heapify(front)
        
        while front:
            current_time, current_point = heapq.heappop(front)
            
            # Update the travel times for neighbors
            for neighbor in self.get_vertex_neighbors(current_point):
                edge_length = np.linalg.norm(self.points[current_point] - self.points[neighbor])
                new_time = current_time + edge_length / self.velocity_field[neighbor]
                
                # If the new time is less than the current estimate, update and push to the heap
                if new_time < travel_times[neighbor]:
                    travel_times[neighbor] = new_time
                    heapq.heappush(front, (new_time, neighbor))
                    
        return travel_times

