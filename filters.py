from networkx import MultiGraph
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

# flag to print deletions
VERBOSE = False

def filterForBike(graph:MultiGraph, verbose):
    to_remove_edges = []

    for u, v, k, data in graph.edges(keys=True, data=True):

        highway = data.get("highway")
        # bicycle = data.get("bicycle") # doesnt return anything for some reason
        access = data.get("access")
        service = data.get("service")

        banned_services = ['parking-aisle', 'alley', 'driveway']
        banned_access = ['private', 'customers', 'military']
        banned_highway = ['service']

        if access in banned_access:
            if VERBOSE: print(f"way removed for banned access\n{data}")
            to_remove_edges.append( { 'u':u, 'v':v, 'k':k } )
        elif service in banned_services:
            if VERBOSE: print(f"way removed for banned service\n{data}")
            to_remove_edges.append( { 'u':u, 'v':v, 'k':k } )
        elif highway in banned_highway:
            if VERBOSE: print(f"way removed for banned higway\n{data}")
            to_remove_edges.append( { 'u':u, 'v':v, 'k':k } )
    
    # remove edges (cant be done in place cause it messes with the dict size)
    for edge in to_remove_edges:
        graph.remove_edge(edge['u'], edge['v'], edge['k'])

    # prune isolated nodes
    to_remove_nodes = []

    for node in graph.nodes:
        if graph.degree(node) == 0:
            to_remove_nodes.append(node)
        
    for node in to_remove_nodes:
        graph.remove_node(node)

    return graph
 



# Bike Map Filtering Rules
# Anything with a bicycle tag of yes, designated, permissive, official, mtb, or MTB is included, or the presence of a bicycle:designated tag.
# Additionally, ways with a highway tag of cycleway are included.
# Ways with bicycle=no but bicycle:conditional are allowed.
# Ways with a bicycle tag of dismount, use_sidepath, private, or no are removed.
# Ways within route=bicycle, mtb and network=lcn, rcn, ncn, icn relations are included.
# Limited access ways are removed: anything with access=private, customers, military.
# If they don't have an allowed bicycle tag, ways with the following highway tags are removed: motorway, motorway_link, steps, stairs, escalator, elevator, construction, proposed, demolished, escape, bus_guideway, sidewalk, crossing, bus_stop, traffic_signals, stop, give_way, milestone, platform, speed_camera, elevator, raceway, rest_area, traffic_island, services, yes, no, drain, street_lamp, razed, corridor, busway, via_ferrata
# These are country-specific, but generally highway=trunk, trunk_link, footway, service, and bridleway are removed (without an allowed bicycle tag)
# Named service roads are included (so long as they don't also have a service=... tag, such as service=parking_aisle)
# Miscellaneous things that are removed: razed, motorroad=yes, golf_cart=yes/designated/private, railway, waterway, route=ferry, mtb:scale=6, indoor=yes, tunnel=yes/building_passage.

# Example
# {'osmid': 1379007000, 'highway': 'service', 'service': 'parking_aisle', 'oneway': False, 'reversed': True, 'length': np.float64(31.210102448560356), 'from': 12769458955, 'to': 12769458921, 'geometry': <LINESTRING (-80.519 43.456, -80.519 43.456)>} None service None
