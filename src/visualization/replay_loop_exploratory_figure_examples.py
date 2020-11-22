__author__ = "Anne Evered"

# %% REQUIRED LIBRARIES
import pandas as pd
import os
from src.visualization.simulation_figures_shared_functions import data_preparation
from src.visualization.simulation_figure_plotly import create_simulation_figure_plotly

# This script uses the plotly versions of simulation output animation/figure
# to explore how we might visualize "replay loop" that Eden (edengh) was working on.
#
# This work is currently in exploratory stage.

# Specify file location and filename
file_location = os.path.join("..", "..", "data", "raw")
filename = "replay-loop.csv"

# Create dataframe and respecify some of the columns so do not have duplicate or ambiguous column names
replay_loop_df = pd.read_csv(os.path.abspath(os.path.join(file_location, filename)))
replay_loop_df.rename(
    columns={
        "temp_basal": "jaeb_temp_basal",
        "suggested_temp_basal_value": "pyloopkit_temp_basal",
    },
    inplace=True,
)
replay_loop_df = data_preparation(replay_loop_df)

# Example where the plot is showing bg as first subplot and the sbr/bolus information for pyloopkit
# as the second subplot and the sbr/bolus information for pyloopkit as the third subplot

traces = [
    {
        0: ["bg_sensor"],
        1: ["sbr", "jaeb_temp_basal_sbr_if_nan", "reported_bolus"],
        2: ["sbr", "suggested_bolus", "pyloopkit_temp_basal_sbr_if_nan"],
    }
]

create_simulation_figure_plotly(
    files_need_loaded=False,
    file_location=file_location,
    file_names=[filename],
    data_frames=[replay_loop_df],
    traces=traces,
    subplots=3,
    time_range=(0, 24),
    main_title="Replay Loop Animation Example",
    subtitle="",
    subplot_titles=[
        "BG Values",
        "Insulin Given (as shown in Jaeb Data)",
        "Insulin Suggested (from PyLoopKit and Bolus Recommendation Tool)",
    ],
    save_fig_path=os.path.join(
        "..", "..", "reports", "figures", "replay_loop_animation_examples"
    ),
    figure_name="plotly_simulation_figure",
    analysis_name="replay_loop",
    animate=True,
)

# Example where the plot is showing bg as first subplot and  then sbr information for both pyloopkit
# and jaeb as the second and the bolus information for pyloopkit and jaeb as the third

traces = [
    {
        0: ["bg_sensor"],
        1: ["sbr", "jaeb_temp_basal_sbr_if_nan", "pyloopkit_temp_basal_sbr_if_nan"],
        2: ["reported_bolus", "suggested_bolus"],
    }
]

create_simulation_figure_plotly(
    files_need_loaded=False,
    file_location=file_location,
    file_names=[filename],
    data_frames=[replay_loop_df],
    traces=traces,
    subplots=3,
    time_range=(0, 24),
    main_title="Replay Loop Animation Example",
    subtitle="",
    subplot_titles=[
        "BG Values",
        "Basal Insulin Given (Jaeb) vs. Suggested (PyLoopKit)",
        "Bolus Reported (Jaeb) vs. Suggested (Bolus Recommendation Tool)",
    ],
    save_fig_path=os.path.join(
        "..", "..", "reports", "figures", "replay_loop_animation_examples"
    ),
    figure_name="plotly_simulation_figure",
    analysis_name="replay_loop",
    animate=True,
)
