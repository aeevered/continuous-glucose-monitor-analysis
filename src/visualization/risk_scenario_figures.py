# %% REQUIRED LIBRARIES
import os
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from celluloid import Camera
from matplotlib.lines import Line2D
from save_view_fig import save_view_fig, save_animation


# This script depends on imagemagick (for saving image). You can download imagemagick here:
# http://www.imagemagick.org/script/download.php

# Create figure
def create_simulation_figure(
        file_location,
        file_names,
        traces,
        subplots,
        time_range=(0, 8),
        title_text="Comparison of BG Simulation Results for Loop vs. No Loop",
        animate_figure=True,
):
    # Load data files
    data_frames = []
    for file in file_names:
        df = data_loading_and_preparation(os.path.abspath(os.path.join(file_location, file)))
        data_frames.append(df)

    # Set fonts
    font = {"size": 8}
    plt.rc("font", **font)

    # Set up figure and axes
    fig, axes = plt.subplots(subplots)
    camera = Camera(fig)
    fig.set_size_inches(10, 3 * subplots)

    # Add layout features
    axes = add_axes(traces=traces, num_subplots=subplots, axes=axes, data_frames=data_frames, time_range=time_range)

    # Add in different animation traces
    if animate_figure:
        time_values = data_frames[0]["hours_post_simulation"]
    else:
        time_values = [max(data_frames[0]["hours_post_simulation"])]

    for t in time_values:
        for index, df in enumerate(data_frames):
            # Create subset of the data
            subset_df = df[df["hours_post_simulation"] < t]
            for subplot in traces[index]:
                for field in traces[index][subplot]:
                    axes = add_plot(subset_df, axes, field=field, subplot=subplot)
        camera.snap()

    # Animate figure
    animation = camera.animate()

    # Create custom legend
    for subplot in range(subplots):
        legend_items = []
        for trace_dict in traces:
            if subplot in trace_dict.keys():
                for field in trace_dict[subplot]:
                    features = get_features_dictionary(field)
                    legend_items.append(
                        Line2D(
                            [0], [0], color=features["color"], label=features["legend_label"],
                            marker=features["marker"],
                            markersize=3, linestyle="None"
                        )
                    )
        axes[subplot].legend(handles=legend_items, loc="upper right")

    # Set layout
    fig.tight_layout()

    # Save figure
    file_path = os.path.join("..", "..", "reports", "figures", "fda-risk-scenarios")

    if animate_figure:
        figure_name = "bg-comparison-animation"
    else:
        figure_name = "bg-comparison-static-fig"

    save_animation(
        animation,
        figure_name=figure_name,
        analysis_name="risk-scenarios",
        save_fig=True,
        save_fig_path=file_path,
        fps=5,
        dpi=100,
    )

    return


def data_loading_and_preparation(filepath):
    sim_df = pd.read_csv(filepath)

    sim_df["five_minute_marks"] = sim_df.index
    sim_df["minutes_post_simulation"] = sim_df["five_minute_marks"].apply(
        lambda x: x * 5
    )
    sim_df["hours_post_simulation"] = sim_df["minutes_post_simulation"] / 60

    sim_df["temp_basal_sbr_if_nan"] = sim_df["temp_basal"].mask(
        pd.isnull, sim_df["sbr"]
    )
    return sim_df

def add_axes(traces, num_subplots, axes, data_frames, time_range=(0, 8)):
    # Get all of the df[field] for the particular subplot
    # Find the min and max of all of those fields
    # See if contains any of the common glucose bg traces, otherwise do insulin axis
    # Call the function for the appropriate axis
    for subplot in range(num_subplots):
        fields_in_subplot = []
        min_value=0
        max_value=0
        for index, traces_per_file in enumerate(traces):
            fields=traces_per_file[subplot]
            fields_in_subplot.append(fields)
            for field in fields:
                if max(data_frames[index][field])> max_value:
                    max_value = max(data_frames[index][field])
                if min(data_frames[index][field])<min_value:
                    min_value = min(data_frames[index][field])
        if "bg" in fields_in_subplot or "bg_sensor" in fields_in_subplot:
            axes = glucose_axis(axes, subplot, time_range, y_range=(min_value, max_value))
        else:
            axes = basal_axis(axes, subplot, time_range, y_range=(min_value, max_value))
    return axes


def glucose_axis(axes, subplot=0, time_range=(0, 8), y_range=(70, 250)):
    axes[subplot].set_ylabel("Glucose (mg/dL)")
    axes[subplot].set_xlabel("Hours")
    axes[subplot].set_xlim(time_range[0], time_range[1])
    axes[subplot].set_ylim(y_range[0] - 30, y_range[1] + 30)
    axes[subplot].set_xticks(
        np.arange(time_range[0], time_range[1] + 1,
            2.0,
        )
    )
    axes[subplot].grid(True)
    return axes


def basal_axis(axes,  subplot=0, time_range=(0, 8), y_range=(0,1)):
    axes[subplot].set_ylabel("Insulin (U or U/hr)")
    axes[subplot].set_xlabel("Hours")
    axes[subplot].set_xlim(time_range[0], time_range[1])
    axes[subplot].set_ylim(y_range[0] - .1, y_range[1] + .1)
    axes[subplot].set_xticks(
        np.arange(time_range[0], time_range[1] + 1, 2.0)
    )
    axes[subplot].grid(True)
    return axes


def add_plot(df, axes, field="bg", subplot=2):
    features_dictionary = get_features_dictionary(field)
    if field == "temp_basal_sbr_if_nan":
        axes[subplot].fill_between(
            df["hours_post_simulation"],
            df[field],
            color=features_dictionary["color"],
            step=features_dictionary["step"],
            alpha=features_dictionary["alpha"],
        )
    if field == "delivered_basal_insulin":
        (markerLines, stemLines, baseLines) = axes[subplot].stem(
            df["hours_post_simulation"],
            df["delivered_basal_insulin"],
            linefmt="#f9706b",
            use_line_collection=True,
        )
        plt.setp(markerLines, color=features_dictionary["color"], markersize=features_dictionary["markersize"])
        plt.setp(stemLines, color=features_dictionary["color"], linewidth=features_dictionary["linewidth"])
        plt.setp(baseLines, color=features_dictionary["color"], linewidth=features_dictionary["linewidth"])
    else:
        axes[subplot].plot(
            df["hours_post_simulation"],
            df[field],
            color=features_dictionary["color"],
            alpha=features_dictionary["alpha"],
            linestyle=features_dictionary["linestyle"],
            linewidth=features_dictionary["linewidth"],
            marker=features_dictionary["marker"],
            markersize=features_dictionary["markersize"],
        )
    return axes


# TODO: have this function pull from design team tools api
# This is where the specific features are pulled from
def get_features_dictionary(field):
    if field == "bg":
        features_dictionary = dict(legend_label="True BG", color="#B1BEFF", alpha=0.5, linestyle="solid", linewidth=1,
                                   marker="o", markersize=2)
    elif field == "bg_sensor":
        features_dictionary = dict(legend_label="Sensor EGV", color="#6AA84F", alpha=0.5, linestyle="solid",
                                   linewidth=1, marker="o", markersize=2)
    elif field == "sbr":
        features_dictionary = dict(legend_label="Scheduled Basal Rate", color="black", alpha=0.5, linestyle="--",
                                   linewidth=1, marker="o", markersize=2)
    elif field == "iob":
        features_dictionary = dict(legend_label="Insulin On Board", color="#744AC2", alpha=0.5, linestyle="solid",
                                   linewidth=1, marker="o", markersize=2)
    elif field == "temp_basal_sbr_if_nan":
        features_dictionary = dict(legend_label="Loop Decision", color="#008ECC", alpha=0.4, linestyle="solid",
                                   step="pre", drawstyle="steps-pre", linewidth=2, marker="o", markersize=2)
    else:  # field == "delivered_basal_insulin":
        features_dictionary = dict(legend_label="Delivered Basal Insulin", color="#f9706b", alpha=0.4,
                                   linestyle="solid",
                                   linewidth=1, marker="o", markersize=2)
    return features_dictionary


def generate_figure_from_user_input():
    # Get user input for features in plot
    file_location = os.path.join("..", "..", "data", "processed")

    num_subplots = int(input("How many subplots in the figure?"))

    file_names = list(input("What file(s) do you want to visualize in the plot (as comma-separated list)?").split(","))
    # risk_scenarios_PyLoopkit v0.1.csv

    traces = []

    for file in file_names:
        subplot_dictionary = dict()
        for subplot in range(num_subplots):
            traces_for_subplot = list(input("What fields do you "
                                            "want to show from " + str(file) + ""
                                                                               ", in subplot " + str(subplot + 1) + ""
                                                                                                                    " (as comma"
                                                                                                                    " separated list)?"))
            subplot_dictionary[subplot] = traces_for_subplot
        traces.append(subplot_dictionary)

    # "bg", "bg_sensor"
    # "sbr", "iob", "delivered_basal_insulin"
    # "sbr"

    hours = int(input("How many hours do you want to show?"))  # 8
    title_text = input("Title of plot:")

    create_simulation_figure(
        file_location=file_location,
        file_names=file_names,
        traces=traces,
        subplots=num_subplots,  # type of axis and reference dataframe
        time_range=[0, hours],
        title_text=title_text,
        animate_figure=False,
    )


#generate_figure_from_user_input()
# Todo: Clean up the formatting functions, add/try different fields
# Todo: Fix the issue with the 1 subplot
# Todo: Add back in the title text
# Todo: Think about what will show for some of the specific risk scenarios

# Create Figures
file_location = os.path.join("..", "..", "data", "processed")
loop_filename = "risk_scenarios_PyLoopkit v0.1.csv"
no_loop_filename = "risk_scenarios_do_nothing.csv"

title_text = (
    "Risk Scenario: Double Carb Entry ( )"
    "\nIn this scenario, a user enters"
    "a carb value into their pump twice (for example after thinking the first entry "
    "\ndid not go through). Loop is then recommending doses based on erroneous carb values. "
    "[SAMPLE TEXT - would review the right"
)

create_simulation_figure(
                        file_location=file_location,
                        file_names=[loop_filename],
                        traces=[{0: ["bg", "bg_sensor"], 1:["sbr", "iob", "delivered_basal_insulin"], 2: ["sbr"]}],
                        subplots=3,
                        time_range=[0, 8],
                        title_text=title_text,
                        animate_figure=False,
)


create_simulation_figure(
                        file_location=file_location,
                        file_names=[no_loop_filename],
                        traces=[{0: ["bg", "bg_sensor"], 1:["iob", "delivered_basal_insulin"], 2: ["temp_basal_sbr_if_nan","sbr"]}],
                        subplots=3,
                        time_range=[0, 8],
                        title_text=title_text,
                        animate_figure=False,
)

'''
create_simulation_figure(
                        file_location=file_location,
                        file_names=[no_loop_filename, loop_filename],
                        traces=[{0: ["bg", "bg_sensor"]}, {1: ["bg", "bg_sensor"]}],  #type of axis and reference dataframe
                        subplots=2,
                        time_range=[0, 8],
                        title_text="Comparison of BG Simulation Results for Loop vs. No Loop",
                        animate_figure=False,
)

create_simulation_figure(
                        file_location=file_location,
                        file_names=[no_loop_filename, loop_filename],
                        traces=[{0: ["bg", "bg_sensor"]}, {1: ["bg", "bg_sensor"]}],  #type of axis and reference dataframe
                        subplots=2,
                        time_range=[0, 8],
                        title_text="Comparison of BG Simulation Results for Loop vs. No Loop",
                        animate_figure=True,
)


create_simulation_figure(
                        file_location=file_location,
                        file_names=[no_loop_filename, loop_filename],
                        traces=[{0: ["bg", "bg_sensor"]}, {0: ["bg", "bg_sensor"]}],  #type of axis and reference dataframe
                        subplots=2,
                        time_range=[0, 8],
                        title_text="Comparison of BG Simulation Results for Loop vs. No Loop",
                        animate_figure=False,
)
'''

# I wonder if for these scenarios, each scenario is going to have different features want
# to emphasize, so for example highlighting the isf might be valuable in one scenario but not the other.

# Could maybe just have an empty subplot and put a text box with all of the elements
# that don't necessarily have to be time-boxed
