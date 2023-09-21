import numpy as np
import copy
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import os.path
from datetime import datetime
import os

# Action part
def circle_radius(coords):

    # Flatten the list and assign to variables (makes code easier to read later)
    x1, y1, x2, y2, x3, y3 = [i for sub in coords for i in sub]

    a = x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2
    b = (x1**2+y1**2)*(y3-y2) + (x2**2+y2**2)*(y1-y3) + (x3**2+y3**2)*(y2-y1)
    c = (x1**2+y1**2)*(x2-x3) + (x2**2+y2**2)*(x3-x1) + (x3**2+y3**2)*(x1-x2)
    d = (x1**2+y1**2)*(x3*y2-x2*y3) + (x2**2+y2**2) * \
        (x1*y3-x3*y1) + (x3**2+y3**2)*(x2*y1-x1*y2)

    # In case a is zero (so radius is infinity)
    try:
        r = abs((b**2+c**2-4*a*d) / abs(4*a**2)) ** 0.5
    except:
        r = 999

    return r


# Returns indexes of next index and index+lookfront
# We need this to calculate the radius for next track section.
def circle_indexes(mylist, index_car, add_index_1=0, add_index_2=0):

    list_len = len(mylist)

    # if index >= list_len:
    #     raise ValueError("Index out of range in circle_indexes()")

    # Use modulo to consider that track is cyclical
    index_1 = (index_car + add_index_1) % list_len
    index_2 = (index_car + add_index_2) % list_len

    return [index_car, index_1, index_2]


def optimal_velocity(track, min_speed, max_speed, look_ahead_points):

    # Calculate the radius for every point of the track
    radius = []
    for i in range(len(track)):
        indexes = circle_indexes(track, i, add_index_1=-1, add_index_2=1)
        coords = [track[indexes[0]],
                  track[indexes[1]], track[indexes[2]]]
        radius.append(circle_radius(coords))

    # Get the max_velocity for the smallest radius
    # That value should multiplied by a constant multiple
    v_min_r = min([r for r in radius if r > 0])**0.5  # Exclude zero-radius points
    constant_multiple = min_speed / v_min_r
    print(f"Constant multiple for optimal speed: {constant_multiple}")

    if look_ahead_points == 0:
        # Get the maximal velocity from radius
        max_velocity = [(constant_multiple * i**0.5) for i in radius]
        # Get velocity from max_velocity (cap at MAX_SPEED)
        velocity = [min(v, max_speed) for v in max_velocity]
        return velocity

    else:
        # Looks at the next n radii of points and takes the minimum
        # goal: reduce lookahead until car crashes bc no time to break
        LOOK_AHEAD_POINTS = look_ahead_points
        radius_lookahead = []
        for i in range(len(radius)):
            next_n_radius = []
            for j in range(LOOK_AHEAD_POINTS+1):
                index = circle_indexes(
                    mylist=radius, index_car=i, add_index_1=j)[1]
                next_n_radius.append(radius[index])
            radius_lookahead.append(min(next_n_radius))
        max_velocity_lookahead = [(constant_multiple * i**0.5)
                                  for i in radius_lookahead]
        velocity_lookahead = [min(v, max_speed)
                              for v in max_velocity_lookahead]
        return velocity_lookahead


# For each point in racing track, check if left curve (returns boolean)
def is_left_curve(coords):

    # Flatten the list and assign to variables (makes code easier to read later)
    x1, y1, x2, y2, x3, y3 = [i for sub in coords for i in sub]

    return ((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)) > 0


# Calculate the distance between 2 points
def dist_2_points(x1, x2, y1, y2):
        return abs(abs(x1-x2)**2 + abs(y1-y2)**2)**0.5


# Optimize part

def dist_2_points(x1, x2, y1, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def x_perc_width(waypoint, perc_width):
    center_x, center_y, inner_x, inner_y, outer_x, outer_y = waypoint
    delta_x = outer_x - inner_x
    delta_y = outer_y - inner_y
    return [
        center_x, center_y,
        inner_x + delta_x * (1 - perc_width) / 2,
        inner_y + delta_y * (1 - perc_width) / 2,
        outer_x - delta_x * (1 - perc_width) / 2,
        outer_y - delta_y * (1 - perc_width) / 2
    ]

# Main Functions

def load_track(track_name):
    return np.load(f"{track_name}.npy")

def reduce_track_width(waypoints, perc_width=0.8):
    return np.array([x_perc_width(waypoint, perc_width) for waypoint in waypoints])

def menger_curvature(pt1, pt2, pt3, atol=1e-3, epsilon=1e-9):
    vec21 = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
    vec23 = np.array([pt3[0] - pt2[0], pt3[1] - pt2[1]])

    norm21 = np.linalg.norm(vec21)
    norm23 = np.linalg.norm(vec23)
    
    if norm21 < epsilon or norm23 < epsilon:
        return 0.0

    dot_product = np.dot(vec21, vec23)
    denominator = norm21 * norm23
    
    # Clamp the value to be in the range [-1, 1] to avoid invalid inputs to arccos
    theta_input = np.clip(dot_product / denominator, -1.0, 1.0)
    
    theta = np.arccos(theta_input)
    
    if np.isclose(theta - np.pi, 0.0, atol=atol):
        theta = 0.0
    
    dist13 = np.linalg.norm(vec21 - vec23)
    
    return 2 * np.sin(theta) / dist13


def improve_race_line(old_line, inner_border, outer_border, xi_iterations=8):
    new_line = copy.deepcopy(old_line)
    ls_inner_border = Polygon(inner_border)
    ls_outer_border = Polygon(outer_border)
    n = len(new_line)

    for i in range(n):
        xi = new_line[i]
        prevprev, prev, nexxt, nexxtnexxt = (i - 2) % n, (i - 1) % n, (i + 1) % n, (i + 2) % n
        ci = menger_curvature(new_line[prev], xi, new_line[nexxt])
        target_ci = (menger_curvature(new_line[prevprev], new_line[prev], xi) + 
                     menger_curvature(xi, new_line[nexxt], new_line[nexxtnexxt])) / 2

        xi_bound1 = xi
        xi_bound2 = np.mean([new_line[nexxt], new_line[prev]], axis=0)
        p_xi = xi

        for _ in range(xi_iterations):
            p_ci = menger_curvature(new_line[prev], p_xi, new_line[nexxt])

            if np.isclose(p_ci, target_ci):
                break

            if p_ci < target_ci:
                xi_bound2 = p_xi
                new_p_xi = np.mean([xi_bound1, p_xi], axis=0)
            else:
                xi_bound1 = p_xi
                new_p_xi = np.mean([xi_bound2, p_xi], axis=0)

            if (Point(new_p_xi).within(ls_inner_border) or not Point(new_p_xi).within(ls_outer_border)):
                continue

            p_xi = new_p_xi

        new_line[i] = p_xi

    return new_line


def plot_track(race_line, inner_border, outer_border):
    fig, ax = plt.subplots()
    ax.plot(*race_line.T, label='Race Line', color='r')
    ax.plot(*inner_border.T, label='Inner Border', color='g')
    ax.plot(*outer_border.T, label='Outer Border', color='b')
    ax.legend()
    plt.show()


# Higher level interface