import glob
import numpy as np
import copy
from shapely.geometry import LineString, Polygon, Point
import matplotlib.pyplot as plt
import os.path
import argparse
from datetime import datetime
import os

# Utility Functions

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


def save_result(array):
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'optimized_waypoints_{timestamp}.npy'
    
    np.save(os.path.join(output_dir, filename), array)
    print(f"Saved successfully as {filename}")



def main(track_file_path, perc_width=0.8, line_iterations=100, xi_iterations=8):
    # Load the track
    waypoints = load_track(track_file_path)

    # Reduce track width
    reduced_waypoints = reduce_track_width(waypoints, perc_width)

    # Get border
    inner_border = waypoints[:, 2:4]
    outer_border = waypoints[:, 4:6]
    center_line = waypoints[:, 0:2]
    
    plot_track(center_line, inner_border, outer_border)
    
    # Improve the race line
    race_line = np.array(reduced_waypoints)[:, 0:2]  # Initialize with center line
    for _ in range(line_iterations):
        race_line = improve_race_line(
            race_line,
            reduced_waypoints[:, 2:4],
            reduced_waypoints[:, 4:6],
            xi_iterations
        )

    # Plot the track
    plot_track(race_line, reduced_waypoints[:, 2:4], reduced_waypoints[:, 4:6])

    # Save the result
    save_result(race_line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize a race line.')
    parser.add_argument('track_file', type=str, help='Path to the .npy file containing the track data.')
    parser.add_argument('--perc_width', type=float, default=0.8, help='Percentage width to reduce the track by.')
    parser.add_argument('--line_iterations', type=int, default=100, help='Number of iterations to improve the line.')
    parser.add_argument('--xi_iterations', type=int, default=8, help='Number of iterations for xi.')
    
    args = parser.parse_args()
    
    main(args.track_file, args.perc_width, args.line_iterations, args.xi_iterations)
