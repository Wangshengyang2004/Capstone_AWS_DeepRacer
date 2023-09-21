import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import math
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from utils import *

# Streamlit app
def streamlit_app():

    st.title('Race Line and Action Space Optimization')

    # Sidebar for user input
    st.sidebar.header('Parameters')
    uploaded_file = st.sidebar.file_uploader("Upload a track file (.npy)", type="npy")
    MIN_SPEED = st.sidebar.slider('Minimum Speed', 0.0, 5.0, 1.3)
    MAX_SPEED = st.sidebar.slider('Maximum Speed', 0.0, 5.0, 4.0)
    LOOK_AHEAD_POINTS = st.sidebar.slider('Look Ahead Points', 0, 20, 5)
    TRACK_NAME = st.sidebar.text_input('Track Name', 'Spain')

    if uploaded_file is not None:
        racing_track = np.load(uploaded_file).tolist()[:-1]

        # Calculate optimal speed
        velocity = optimal_velocity(track=racing_track,
                                    min_speed=MIN_SPEED, max_speed=MAX_SPEED, look_ahead_points=LOOK_AHEAD_POINTS)

        # Plot Velocity
        x = [i[0] for i in racing_track]
        y = [i[1] for i in racing_track]
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=x, y=y, hue=velocity, palette="vlag", ax=ax1)
        ax1.set_title(f"With lookahead: {LOOK_AHEAD_POINTS}")
        st.pyplot(fig1)

        # ... (You can include more processing and plotting logic here) ...


import streamlit as st

# ... Your function definitions remain the same ...

# Streamlit app
def streamlit_app():
    st.title('Race Line Optimization')

    # User input
    uploaded_file = st.file_uploader("Choose a track file (.npy)", type="npy")
    perc_width = st.slider('Percentage Width to Reduce the Track By', 0.0, 1.0, 0.8)
    line_iterations = st.number_input('Number of iterations to improve the line', 1, 500, 100)
    xi_iterations = st.number_input('Number of iterations for xi', 1, 20, 8)

    if uploaded_file is not None:
        # Load track
        waypoints = np.load(uploaded_file)

        # Reduce track width
        reduced_waypoints = reduce_track_width(waypoints, perc_width)

        # Get borders
        inner_border = waypoints[:, 2:4]
        outer_border = waypoints[:, 4:6]
        center_line = waypoints[:, 0:2]

        # Plot initial track
        st.write("Initial Track:")
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

        # Plot optimized track
        st.write("Optimized Track:")
        plot_track(race_line, reduced_waypoints[:, 2:4], reduced_waypoints[:, 4:6])


# Main entry point
if __name__ == '__main__':
    streamlit_app()
