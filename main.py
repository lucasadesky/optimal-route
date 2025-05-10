import osmnx as ox
import matplotlib.pyplot as plt
import dataframe_image as dfi
import numpy as np
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.widgets import Button
import networkx as nx

import gpxpy
import gpxpy.gpx

from shapely.geometry import Polygon as SPoly

from utils import assign_edge_colors
from filters import filterForBike

VERBOSE = True

filename = "route"

gpx = gpxpy.gpx.GPX()

# Create first track in our GPX:
gpx_track = gpxpy.gpx.GPXTrack()
gpx.tracks.append(gpx_track)

# Create first segment in our GPX track:
gpx_segment = gpxpy.gpx.GPXTrackSegment()
gpx_track.segments.append(gpx_segment)


# ############## Functions

def toggle_recording(event):
    global recording
    recording = not recording
    record_button.label.set_text('Stop Drawing' if recording else 'Draw Border')
    fig.canvas.draw_idle()

# Function to calculate the direction vector between two points
def direction_vector(p1, p2):
    return np.array([p2['x'] - p1['x'], p2['y'] - p1['y']])

# Function to calculate the angle between two vectors
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Ensure the cosine is within the range [-1, 1] due to precision errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Angle in radians and then converted to degrees
    angle = np.arccos(cos_angle) * (180 / np.pi)
    return angle

# Function to calculate cross product (to determine turn direction)
def cross_product(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def rotate_vector(vec, angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians),  np.cos(angle_radians)]
    ])
    return rotation_matrix @ vec

###########################


# Color Settings
BG_COLOR = (0,0,0)
NODE_COLOR = "w"
EDGE_COLORS = ["y", "b", "m", "c"]

# Size
NODE_SIZE = 15
EDGE_LW = 3

# Polygon drawing state
polygon_points = []
polygon_line = None
completed_polygon = None
polygon_created = False
recording = False
starting_node_idx = 0
starting_marker = None
starting_node_id = None

# graph = ox.graph_from_place("Vernon, BC, Canada", simplify=False, network_type="bike")
# waterloo: 43.4593402, -80.5284608

directed_graph = ox.graph_from_point((43.4593402, -80.5284608), 400, simplify=True, network_type='bike')
unfiltered_graph = ox.convert.to_undirected(directed_graph)

filtered_graph = filterForBike(unfiltered_graph, verbose=False)

# Assign colors to edges
edge_colors = assign_edge_colors(filtered_graph, EDGE_COLORS)

# Plot the graph
fig, ax = ox.plot_graph(
    filtered_graph,
    node_color=NODE_COLOR,
    node_size=NODE_SIZE,
    edge_linewidth=EDGE_LW,
    edge_color=list(edge_colors.values()),
    bgcolor=BG_COLOR,
    show=False,
    save=False,
    close=False
)

# Define function to reset/complete polygon
def complete_polygon():
    global polygon_points, polygon_line, completed_polygon, polygon_created
    
    if len(polygon_points) >= 3:
        # Convert points to numpy array for matplotlib
        poly_array = np.array(polygon_points)
        
        # Create a polygon patch
        poly = Polygon(poly_array, 
                        facecolor='yellow', 
                        alpha=0.2, 
                        edgecolor='yellow',
                        linewidth=2)
        
        completed_polygon = SPoly(poly_array)  # shapely polygon for using to collect roads

        polygon_created = True
        
        # Add to plot
        ax.add_patch(poly)
        
        # Print polygon coordinates
        print("\nPolygon completed with the following coordinates:")
        for i, (x, y) in enumerate(polygon_points):
            print(f"  Point {i+1}: Longitude: {x:.6f}, Latitude: {y:.6f}")
    
    # Reset for next polygon
    if polygon_line:
        polygon_line.remove()
    polygon_points = []
    polygon_line = None
    fig.canvas.draw_idle()

# Define click handler function
def on_click(event):
    global polygon_points, polygon_line, recording, polygon_created

    if not recording or polygon_created:
        return
    
    if event.xdata is not None and event.ydata is not None:
        # Print the coordinates
        print(f"Clicked coordinates: Longitude: {event.xdata:.6f}, Latitude: {event.ydata:.6f}")
        
        # Check if we're completing a polygon (right-click)
        if event.button == 3:  # Right mouse button
            complete_polygon()
            return
            
        # Add point to the polygon
        polygon_points.append((event.xdata, event.ydata))
        
        # Draw the point
        ax.plot(event.xdata, event.ydata, 'ro', markersize=5)
        
        # If we have at least 2 points, draw/update the line
        if len(polygon_points) >= 2:
            xs, ys = zip(*polygon_points)
            
            # If line exists, remove it before drawing the new one
            if polygon_line:
                polygon_line.remove()
            
            # Draw new line connecting all points
            polygon_line, = ax.plot(xs, ys, 'r-', linewidth=2)
            
        # Update the display
        fig.canvas.draw_idle()

#Define key handler function
def on_key(event):

    # print(event.key)
        
    global polygon_created, polygon_points, completed_polygon, starting_node_idx, starting_marker, starting_node_id
        
    if polygon_created:
        print('key pressed')

        # Retrieve the graph based on the completed polygon
        temp_polygon_graph = ox.graph_from_polygon(completed_polygon, network_type='bike', simplify=True)
        temp_polygon_graph = ox.convert.to_undirected(temp_polygon_graph)

        tempG = filterForBike(temp_polygon_graph, False)

        nodes_list = []

        for node in tempG.nodes:
            # print(node)
            nodes_list.append(node)

        # starting_node_idx = 0
        len_list = len(nodes_list) -1

        if event.key == ' ' or event.key == 'right':
            if starting_node_idx + 1 < len_list:
                starting_node_idx += 1
            else:
                starting_node_idx = 0
        elif event.key == 'left':
            if starting_node_idx -1 > 0:
                starting_node_idx -= 1
            else:
                starting_node_idx = len_list

        
        print(tempG.nodes[nodes_list[starting_node_idx]])

        # Before drawing a new marker, remove the previous one
        if starting_marker is not None:
            starting_marker.remove()

        # Draw the new marker and store the reference
        starting_marker = ax.plot(
            tempG.nodes[nodes_list[starting_node_idx]]['x'],
            tempG.nodes[nodes_list[starting_node_idx]]['y'],
            'go', markersize=10
            )[0]
        starting_node_id = nodes_list[starting_node_idx]
        # ax.plot(tempG.nodes[nodes_list[starting_node_idx]]['x'], tempG.nodes[nodes_list[starting_node_idx]]['y'], 'go', markersize=10)
    
        # Update the display
        fig.canvas.draw_idle()
    else:
        return

# Add text for instructions
plt.figtext(0.5, 0.01, "Left-click to add points, Right-click to complete polygon", 
           ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.7, "pad":5})

# Add an arrow for more clarity
arrow = FancyArrowPatch((0.5, 0.05), (0.5, 0.1), mutation_scale=20, color='yellow', arrowstyle='->')
ax.add_patch(arrow)

# Connect the click event to the figure
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)

# Add a button to toggle polygon recording
record_button_ax = plt.axes([0.8, 0.01, 0.15, 0.05])
record_button = Button(record_button_ax, 'Draw Border' if not recording else 'Stop Drawing')

save_button_ax = plt.axes([0.6, 0.01, 0.15, 0.05])
save_button = Button(save_button_ax, 'Generate Path')



def save_polygon(event):
    global recording
    recording = False
    # plt.close()
    
    # Retrieve the graph based on the completed polygon
    osmnx_polygon_graph = ox.graph_from_polygon(completed_polygon, network_type='bike', simplify=True)
    osmnx_polygon_graph = ox.convert.to_undirected(osmnx_polygon_graph)

    osmnx_polygon_graph_filtered = filterForBike(osmnx_polygon_graph, False)
    
    print(f'Number of roads selected: {len(osmnx_polygon_graph_filtered.edges)}')  # Number of edges (roads) in the graph
    for u, v, data in osmnx_polygon_graph_filtered.edges(data=True):  # Iterate over edges with data
        road_name = data.get('name', 'No name available')  # Get road name (if available)
        if VERBOSE: print(f"  Road name: {road_name}")

    find_route(osmnx_polygon_graph_filtered)

    

def find_route(osmnx_selected_graph):
    global starting_node_id
    
    dead_end_nodes = []

    # get the odd nodes
    for node_id in osmnx_selected_graph.nodes:
        degree = osmnx_selected_graph.to_undirected().degree(node_id)
        if degree == 1:
            dead_end_nodes.append(node_id)

    print(f'Dead end nodes: {len(dead_end_nodes)}')
    print(f'Total nodes: {len(osmnx_selected_graph.nodes)}')

    dead_end_pairs = []

    # dead end nodes' edges are always doubled
    for node in dead_end_nodes:
        dead_end_pairs.append( (node, list(osmnx_selected_graph.neighbors(node))[0]) )

    print(dead_end_pairs)

    for node1, node2 in dead_end_pairs:
        print(f"Matched {node1} with {node2}")
        path = nx.shortest_path(osmnx_selected_graph, node1, node2, weight='length')
        for u, v in zip(path[:-1], path[1:]):
            # Grab existing edges between u and v
            existing_edges = osmnx_selected_graph.get_edge_data(u, v, default={})

            # Generate a new key that doesn't collide
            existing_keys = existing_edges.keys()
            new_key = max(existing_keys, default=-1) + 1

            # Use attributes from the first existing edge if available
            base_attrs = next(iter(existing_edges.values()), {})
            attrs = dict(base_attrs)  # Copy to avoid modifying original
            attrs['augmented'] = True

            osmnx_selected_graph.add_edge(u, v)

    
    odd_degree_nodes = []
    
    # get the odd nodes
    for node_id in osmnx_selected_graph.nodes:
        degree = osmnx_selected_graph.degree(node_id)
        if degree % 2 != 0:
            odd_degree_nodes.append(node_id)

    print(f'odd degree nodes: {odd_degree_nodes}')

    odd_degree_pairs = []

    # pair the odd nodes
    odd_degree_connectivity = compute_distances_graph(odd_degree_nodes, osmnx_selected_graph)
    if len(odd_degree_nodes) >= 2:  # Only call min_weight_matching if we have at least 2 odd nodes
        for i in nx.algorithms.matching.min_weight_matching(odd_degree_connectivity):
            odd_degree_pairs.append(i)

    # add the new pairs to the graph
    for node1, node2 in odd_degree_pairs:
        print(f"Matched {node1} with {node2}")
        path = nx.shortest_path(osmnx_selected_graph, node1, node2, weight='length')
        for u, v in zip(path[:-1], path[1:]):
            # Grab existing edges between u and v
            existing_edges = osmnx_selected_graph.get_edge_data(u, v, default={})

            # Generate a new key that doesn't collide
            existing_keys = existing_edges.keys()
            new_key = max(existing_keys, default=-1) + 1

            # Use attributes from the first existing edge if available
            base_attrs = next(iter(existing_edges.values()), {})
            attrs = dict(base_attrs)  # Copy to avoid modifying original
            attrs['augmented'] = True

            osmnx_selected_graph.add_edge(u, v)

    print(f'{"is connected" if nx.is_connected(osmnx_selected_graph) else "is not connected"}')

    # Check that all nodes have even degree
    for node in osmnx_selected_graph.nodes():
        print(f"Node {node} has degree {osmnx_selected_graph.degree(node)}")

    nodes_list = []

    # HERE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    starting_node = starting_node_id

    circuit = list(nx.eulerian_circuit(osmnx_selected_graph, source=starting_node))
    print(f'Segments in loop: {len(circuit)}')

    #  loop through the circuit and get the latlon data to add to the gpx file
    for edge in circuit:
        # get the edge from the graph
        edge_data = osmnx_selected_graph.get_edge_data(edge[0], edge[1])

        # some roads with fine detail are travelled backwards so need their points reversed
        backwards = False
        if edge_data[0]['from'] != edge[0]:
            backwards = True
        
        # store the points for curved roads so they arent just straight lines
        points = []

        # if there's road curvature get it from the edge data
        if 'geometry' in edge_data[0]:
            line = edge_data[0]['geometry']

            # cast to list and reverse if traveresed backwards
            points = list(line.coords)
            if backwards:
                points = points[::-1]

        # if road data not found just use linear approx from start and end points
        else:
            points = [(osmnx_selected_graph.nodes[edge[0]]['x'], osmnx_selected_graph.nodes[edge[0]]['y']), (osmnx_selected_graph.nodes[edge[1]]['x'], osmnx_selected_graph.nodes[edge[1]]['y'])]

        if VERBOSE: print(f'Edge: {edge} | Points: {points}')
        for point in points:
            gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(point[1], point[0]))

        # print(edge)
        nodes_list.append(edge[0])

    # add the last edge
    nodes_list.append(circuit[-1][1])

    with open(f"{filename}.gpx", 'w') as file:
        file.write(gpx.to_xml())

    print(f"Written to {filename}.gpx")


    # not drawing for now
    # draw_route(nodes_list, directed_graph, ax)


    

def compute_distances_graph(odd_nodes, osmnx_selected_graph):
    NX_Weighted_Connectivity_Graph = nx.Graph()

    for i, u in enumerate(odd_nodes):
        for j in range(i + 1, len(odd_nodes)):
            v = odd_nodes[j]
            try:
                length = nx.shortest_path_length(osmnx_selected_graph, u, v, weight='length')
                NX_Weighted_Connectivity_Graph.add_edge(u, v, weight=length)
            except nx.NetworkXNoPath:
                pass  # or handle if nodes are disconnected 

    # print(nx.to_dict_of_dicts(NX_Weighted_Connectivity_Graph))
    return NX_Weighted_Connectivity_Graph


def draw_route(nodes_list, directed_graph, ax=None):
    # Get the base plot but don't show it yet
    fig, new_ax = ox.plot_graph_route(
        directed_graph,
        nodes_list,
        route_color='purple',
        route_linewidth=4,
        node_size=0,
        show=False,
        close=False,
        ax=ax
    )

    node_list_len = len(nodes_list)

    for i in range(1, node_list_len - 1):

        current_node = nodes_list[i]
        next_node = nodes_list[i+1]
        prev_node = nodes_list[i-1]

        incoming = direction_vector(directed_graph.nodes[nodes_list[i-1]], directed_graph.nodes[nodes_list[i]])
        outgoing = direction_vector(directed_graph.nodes[nodes_list[i]], directed_graph.nodes[nodes_list[i+1]])

        horizontal_reference = direction_vector({'x':0, 'y':0}, {'x': 1, 'y': 0})

        # Calculate the angle between vectors
        angle = calculate_angle(incoming, outgoing)
        incoming_angle = calculate_angle(incoming, horizontal_reference) % 180
        print(f'Incoming angle: {incoming_angle}')

        # Determine turn direction using cross product
        cp = cross_product(incoming, outgoing)

        current_coords = (directed_graph.nodes[current_node]['x'], directed_graph.nodes[current_node]['y'])
        next_coords = (directed_graph.nodes[next_node]['x'], directed_graph.nodes[next_node]['y'])
        prev_coords = (directed_graph.nodes[prev_node]['x'], directed_graph.nodes[prev_node]['y'])

        bisector = (-incoming / np.linalg.norm(incoming)) + (outgoing / np.linalg.norm(outgoing))
        start_sector = np.linalg.norm(incoming)
        end_sector = np.linalg.norm(bisector / np.linalg.norm(bisector)) + (outgoing / np.linalg.norm(outgoing))

        turn_direction = None

        # deltas
        dx = current_coords[0] - prev_coords[0]
        dy = current_coords[1] - prev_coords[1]

        # binaries
        bx = 1 if (current_coords[0] > prev_coords[0]) else -1
        by = 1 if (current_coords[1] > prev_coords[1]) else -1

        # set the opposite binary to zero so we dont get 45degree lines
        if abs(dx) > abs(dy):
            by = 0
        else:
            bx = 0

        # turn direction from crossprod
        if cp > 0:
            turn_direction = 1  # Left turn
        else:
            turn_direction = -1  # Right turn

        # arrow dims
        spacing = 0.0001 # perp dist to road
        tail = 0.00025 # arrow tail length
        
        # Draw the arrow based on the angle and turn type
        if angle < 30:
            # Straight arrow (no curvature)
            arrow = FancyArrowPatch(
                (current_coords[0] - abs(bx)*tail + spacing*by, current_coords[1] + spacing*bx - abs(by)*tail), 
                (current_coords[0] + abs(bx)*tail + spacing*by, current_coords[1] + spacing*bx + abs(by)*tail),
                arrowstyle=f" {'<-' if bx > 0 or by < 0 else '->'}", 
                mutation_scale=15, 
                color='yellow', 
                linewidth=2
            )
        elif 60 < angle < 120:

            if turn_direction > 0:
                # Left turn

                start = current_coords + spacing*start_sector
                end = current_coords + spacing*end_sector

                arrow = FancyArrowPatch(
                    (start), 
                    (end), 
                    arrowstyle='->', 
                    connectionstyle=f"arc3,rad={0.3}",  # Adjust curvature
                    mutation_scale=15, 
                    color='red', 
                    linewidth=2
                )
            else:
                # Right turn

                # these arent vectors!!!!! you migh just wanna define vector class and make up ur own math
                start = incoming + rotate_vector(incoming, angle/5)
                end = outgoing + rotate_vector(outgoing, angle*4/5)

                new_ax.plot(start[0], start[1], 'go', markersize=5)
                new_ax.plot(end[0], end[1], 'ro', markersize=5)

                arrow = FancyArrowPatch(
                    (start), 
                    (end), 
                    arrowstyle='->', 
                    connectionstyle=f"angle,angleA={0}, angleB={90}, rad={5}",  # Adjust curvature
                    mutation_scale=10, 
                    color='white', 
                    linewidth=2
                )
        elif angle > 150:
            # U-turn
            arrow = FancyArrowPatch(
                (current_coords[0] - 0.0002, current_coords[1]),  # U-turn pointing back
                (current_coords[0] + 0.0002, current_coords[1]), 
                arrowstyle='->', 
                connectionstyle=f"arc3,rad={-1}",  # Larger curvature for U-turn
                mutation_scale=20, 
                color='purple', 
                linewidth=2
            )

        # Add the arrow to the plot
            ax.add_patch(arrow)

    plt.show()


record_button.on_clicked(toggle_recording)
save_button.on_clicked(save_polygon)

plt.show()  # Keep the plot window open

