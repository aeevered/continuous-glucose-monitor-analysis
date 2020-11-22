__author__ = "Anne Evered"

# %% REQUIRED LIBRARIES
import os
from src.visualization.simulation_figure_plotly import create_simulation_figure_plotly
from src.visualization.simulation_figure_matplotlib import (
    create_simulation_figure_matplotlib,
    generate_figure_from_user_input,
)
from src.visualization.simulation_figures_shared_functions import (
    data_loading_and_preparation,
    get_features_dictionary,
)

# This script provides examples of the matplotlib and plotly versions of simulation
# output animation/figure for the purpose of testing,
# illustrating how these functions work and/or demoing.

############## Simulation Animation Plotly Examples #################

# Specify file location of where want to save examples to
save_fig_path = os.path.join(
    "..",
    "..",
    "reports",
    "figures",
    "simulation_figure_examples",
    "plotly_simulation_figures",
)

# Specify file location and filename of files to include in visualization
file_location = os.path.join("..", "..", "data", "raw")
loop_filename = "risk_scenarios_PyLoopkit v0.1.csv"
no_loop_filename = "risk_scenarios_do_nothing.csv"


# Example of just showing data from the no loop example file
traces = [{0: ["bg", "bg_sensor"], 1: ["iob"], 2: ["sbr"]}]

create_simulation_figure_plotly(
    files_need_loaded=True,
    file_location=file_location,
    file_names=[no_loop_filename],
    traces=traces,
    subplots=3,
    time_range=(0, 8),
    main_title="Risk Scenario",
    subtitle="",
    subplot_titles=[
        "BG Values",
        "Insulin On-Board",
        "Scheduled Basal Rate",
    ],
    save_fig_path=save_fig_path,
    figure_name="plotly_simulation_figure",
    analysis_name="risk_scenarios",
    animate=True,
)

# Example of just showing data from the loop example file
traces = [{0: ["bg", "bg_sensor"], 1: ["iob"], 2: ["sbr", "temp_basal_sbr_if_nan"]}]

create_simulation_figure_plotly(
    files_need_loaded=True,
    file_location=file_location,
    file_names=[loop_filename],
    traces=traces,
    subplots=3,
    time_range=(0, 8),
    subtitle="",
    main_title="Risk Scenario",
    subplot_titles=[
        "BG Values",
        "Insulin On-Board",
        "Scheduled Basal Rate and Loop Decisions",
    ],
    save_fig_path=save_fig_path,
    figure_name="plotly_simulation_figure",
    analysis_name="risk_scenarios",
    animate=True,
)

# Example of showing data from the loop example file and the no loop example file in one plot

# List of dictionaries (one dictionary per file)
traces = [
    {0: ["bg", "bg_sensor"], 1: ["sbr"]},
    {2: ["bg", "bg_sensor"], 3: ["sbr", "temp_basal_sbr_if_nan"]},
]

create_simulation_figure_plotly(
    files_need_loaded=True,
    file_location=file_location,
    file_names=[no_loop_filename, loop_filename],
    traces=traces,
    subplots=4,
    time_range=(0, 8),
    subtitle="",
    main_title="Risk Scenario",
    subplot_titles=[
        "BG Values (No Loop)",
        "Scheduled Basal Rate (No Loop)",
        "BG Values (With Loop)",
        "Scheduled Basal Rate and Loop Decisions",
    ],
    save_fig_path=save_fig_path,
    figure_name="plotly_simulation_figure",
    analysis_name="risk_scenarios",
    animate=True,
)

############## Simulation Animation Matplotlib Examples #################


def simulation_description_title(
    file, simulation_description, simulation_name, fields_for_title
):
    """
    For the purposes of these examples, this function creates a title from various elements
    like the fields, simulation name, and file.

    Parameters
    ----------
    file: str
        filename for title
    simulation_description: str
        simulation_description for title
    simulation_name: str
        simulation_name for title
    fields_for_title: str
        fields to include in title

    Returns
    -------

    """
    df = data_loading_and_preparation(
        os.path.abspath(os.path.join(file_location, file))
    )

    # Create title
    title_text = (
        "\n" + simulation_name + "\n\n" + simulation_description + "\nScenario Details:"
    )

    # Add to title the fields and values for the particular fields passed in
    for field in fields_for_title:
        title_text = (
            title_text
            + "\n"
            + get_features_dictionary(field)["legend_label"]
            + ": "
            + str(df.iloc[0][field])
        )

    return title_text


file_location = os.path.join("..", "..", "data", "raw")
loop_filename = "risk_scenarios_PyLoopkit v0.1.csv"
no_loop_filename = "risk_scenarios_do_nothing.csv"

simulation_name = "Risk Scenario: Double Carb Entry"
simulation_description = (
    "In this scenario, a user enters"
    "a carb value into their pump twice (for example after thinking the first entry "
    "\ndid not go through). Loop is then recommending doses based on erroneous carb values."
)
fields = [
    "sbr",
    "pump_sbr",
    "isf",
    "pump_isf",
    "cir",
    "pump_cir",
    "true_bolus",
    "reported_bolus",
    "true_carb_value",
    "reported_carb_value",
]

main_title = simulation_description_title(
    loop_filename, simulation_description, simulation_name, fields_for_title=fields
)

# Single visualization showing elements without loop
create_simulation_figure_matplotlib(
    file_location=file_location,
    file_names=[no_loop_filename],
    traces=[
        {0: ["bg", "bg_sensor"], 1: ["iob"], 2: ["sbr", "delivered_basal_insulin"]}
    ],
    subplots=3,
    time_range=[0, 8],
    main_title=main_title,
    subplot_titles=[
        "BG Values",
        "Insulin On-Board",
        "Scheduled Basal Rate and Delivered Basal Insulin",
    ],
    animate_figure=False,
    figure_name="simulaton_no_loop",
    save_fig_path=save_fig_path,
)

# Single visualization showing elements with loop
create_simulation_figure_matplotlib(
    file_location=file_location,
    file_names=[loop_filename],
    traces=[
        {
            0: ["bg", "bg_sensor"],
            1: ["iob", "reported_bolus", "true_bolus"],
            2: ["temp_basal_sbr_if_nan", "sbr", "delivered_basal_insulin"],
        }
    ],
    subplots=3,
    time_range=[0, 8],
    main_title=main_title,
    subplot_titles=[
        "BG Values",
        "Insulin On-Board",
        "Loop Decisions, Scheduled Basal Rate, and Delivered Basal Insulin",
    ],
    animate_figure=False,
    figure_name="simulaton_with_loop",
    save_fig_path=save_fig_path,
)

# Comparison of loop to not loop in one visualization (both insulin and bg)
create_simulation_figure_matplotlib(
    file_location=file_location,
    file_names=[no_loop_filename, loop_filename],
    traces=[{0: ["bg", "bg_sensor"]}, {1: ["bg", "bg_sensor"]}],
    subplots=2,
    time_range=[0, 8],
    main_title="Comparison of BG Simulation Results for Loop vs. No Loop",
    subplot_titles=[],
    animate_figure=False,
    figure_name="loop_bg_comparison",
    save_fig_path=save_fig_path,
)

# Comparison of loop to not loop in one visualization (just bg)
create_simulation_figure_matplotlib(
    file_location=file_location,
    file_names=[no_loop_filename, loop_filename],
    traces=[
        {0: ["bg", "bg_sensor"], 1: ["sbr", "delivered_basal_insulin"]},
        {
            2: ["bg", "bg_sensor"],
            3: ["temp_basal_sbr_if_nan", "sbr", "delivered_basal_insulin"],
        },
    ],
    subplots=4,
    time_range=[0, 8],
    main_title="Comparison of Simulation Results for Loop vs. No Loop",
    subplot_titles=[
        "BG Values without Loop",
        "Basal Rate without Loop",
        "BG Values with Loop",
        "Basal Rate with Loop",
    ],
    animate_figure=False,
    figure_name="loop_bg_insulin, comparison",
    save_fig_path=save_fig_path,
)


# Example from user input
# generate_figure_from_user_input()
