import glob
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import LinearRing, LineString
import pandas as pd
import matplotlib.pyplot as plt
import os.path

# Conveniently list available tracks to analyze
available_track_files = glob.glob("**.npy")
available_track_names = list(map(lambda x: os.path.basename(x).split('.npy')[0], available_track_files))
available_track_names

# Replace the name here with the track to analyze
TRACK_NAME = available_track_names[1]

# Load the center, inner, outer waypoints
waypoints = np.load("%s.npy" % TRACK_NAME)

# # Convert to Shapely objects
center_line = waypoints[:, 0:2]
inner_border = waypoints[:, 2:4]
outer_border = waypoints[:, 4:6]

l_center_line = LineString(center_line)
l_inner_border = LineString(inner_border)
l_outer_border = LineString(outer_border)
# Get coordinates from LineString objects
outer_coords = np.array(l_outer_border.coords)
inner_coords = np.array(l_inner_border.coords)

# Create the Polygon
road_poly = Polygon(np.vstack((outer_coords, np.flipud(inner_coords))))

# Check if the center line is a loop/ring
print("Is loop/ring? ", l_center_line.is_ring)

# Own code: Reduce width of track

def dist_2_points(x1, x2, y1, y2):
        return abs(abs(x1-x2)**2 + abs(y1-y2)**2)**0.5
    
def x_perc_width(waypoint, perc_width):
    
    center_x, center_y, inner_x, inner_y, outer_x, outer_y = waypoint
    
    width = dist_2_points(inner_x, outer_x, inner_y, outer_y)
    
    delta_x = outer_x-inner_x
    delta_y = outer_y-inner_y
    
    inner_x_new = inner_x + delta_x/2 * (1-perc_width)
    outer_x_new = outer_x - delta_x/2 * (1-perc_width)
    inner_y_new = inner_y + delta_y/2 * (1-perc_width)
    outer_y_new = outer_y - delta_y/2 * (1-perc_width)
    
    return [center_x, center_y, inner_x_new, inner_y_new, outer_x_new, outer_y_new]

PERC_WIDTH = 0.8
waypoints_new = [x_perc_width(waypoint, perc_width=PERC_WIDTH) for waypoint in waypoints]
waypoints_new = np.asarray(waypoints_new)

# Convert to Shapely objects
inner_border_new = waypoints_new[:,2:4]
outer_border_new = waypoints_new[:,4:6]
l_inner_border_new = LineString(inner_border_new)
l_outer_border_new = LineString(outer_border_new)
# Get coordinates from LineString objects
outer_coords_new = np.array(l_outer_border_new.coords)
inner_coords_new = np.array(l_inner_border_new.coords)

# Create the Polygon
road_poly_new = Polygon(np.vstack((outer_coords_new, np.flipud(inner_coords_new))))

# Check if the center line is a loop/ring
print("Is loop/ring? ", l_center_line.is_ring)
