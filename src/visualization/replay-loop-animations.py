
import pandas as pd
import os
from risk_scenario_figures_shared_functions import data_preparation
from risk_scenario_figures_plotly import create_simulation_figure_plotly
import numpy as np

file_location = os.path.join("..", "..", "data", "processed")
filename = "loop_replay-nogit.csv"

replay_loop_df = pd.read_csv(os.path.abspath(os.path.join(file_location, filename)))
replay_loop_df.rename(columns={'temp_basal':'jaeb_temp_basal', 'suggested_temp_basal_value':'pyloopkit_temp_basal'}, inplace=True)
replay_loop_df = data_preparation(replay_loop_df)

print(replay_loop_df[replay_loop_df["reported_bolus"]!=0]["reported_bolus"])

traces = [{0: ["bg_sensor"], 1: ["sbr", "jaeb_temp_basal_sbr_if_nan", "reported_bolus"], 2: ["sbr","suggested_bolus", "pyloopkit_temp_basal_sbr_if_nan"]}]


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
    subplot_titles=["BG Values", "Insulin Given (as shown in Jaeb Data)", "Insulin Suggested (from PyLoopKit and Bolus Recommendation Tool)"],
    save_fig_path=os.path.join("..", "..", "reports", "figures", "fda-risk-scenarios"),
    figure_name="plotly_simulation_figure",
    analysis_name="risk_scenarios",
    animate=True,
)

traces = [{0: ["bg_sensor"], 1: ["sbr", "jaeb_temp_basal_sbr_if_nan", "pyloopkit_temp_basal_sbr_if_nan"], 2:["reported_bolus", "suggested_bolus"]}]

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
    subplot_titles=["BG Values", "Basal Insulin Given (Jaeb) vs. Suggested (PyLoopKit)", "Bolus Reported (Jaeb) vs. Suggested (Bolus Recommendation Tool)"],
    save_fig_path=os.path.join("..", "..", "reports", "figures", "fda-risk-scenarios"),
    figure_name="plotly_simulation_figure",
    analysis_name="risk_scenarios",
    animate=True,
)