__author__ = "Anne Evered"

# %% REQUIRED LIBRARIES
import os
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from celluloid import Camera
from matplotlib.lines import Line2D
from src.visualization.save_view_fig import save_view_fig, save_animation
from src.visualization.simulation_figures_shared_functions import (
    data_loading_and_preparation,
    get_features_dictionary,
)

# These functions are for creating an animation or static figure of simulation
# output in Matplotlib that roughly follows Tidepool design color scheme, etc. for
# different simulation elements (bg, temp_basal, sbr, etc.).

# This animation depends on imagemagick (for saving image). You can download imagemagick here:
# http://www.imagemagick.org/script/download.php


# Create figure
def create_simulation_figure_matplotlib(
    file_location,
    file_names,
    traces,
    subplots,
    time_range=(0, 8),
    main_title="",
    subplot_titles=[],
    animate_figure=True,
    figure_name="simulation_figure",
    save_fig_path=os.path.join("..", "..", "reports", "figures", "simulation_figure_examples")
):
    """

    Parameters
    ----------
    file_location
    file_names
    traces
    subplots
    time_range
    main_title
    subplot_titles
    animate_figure
    figure_name
    save_fig_path

    Returns
    -------

    """
    # Load data files
    data_frames = []
    for file in file_names:
        df = data_loading_and_preparation(
            os.path.abspath(os.path.join(file_location, file))
        )
        data_frames.append(df)

    # Set fonts
    font = {"size": 8}
    plt.rc("font", **font)

    # Set up figure and axes
    fig, axes = plt.subplots(subplots, constrained_layout=True)

    camera = Camera(fig)
    fig.set_size_inches(10, 3 * subplots)
    fig.suptitle(
        main_title, fontsize=10
    )  # fontsize=11, x=.05, horizontalalignment="left")

    # Add layout features
    axes = add_axes(
        traces=traces,
        num_subplots=subplots,
        axes=axes,
        data_frames=data_frames,
        time_range=time_range,
        subplot_titles=subplot_titles,
    )

    # Create custom legend
    axes = add_default_legend(axes, subplots, traces)

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
                    axes = add_plot(
                        subset_df, subplots, axes, field=field, subplot=subplot
                    )
        camera.snap()

    # Animate figure
    animation = camera.animate()

    # Set layout
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95]) #[0, 0.03, 1, 0.85])

    # Save figure

    if animate_figure:
        figure_name = figure_name + "-animation"
    else:
        figure_name = figure_name + "-static"

    save_animation(
        animation,
        figure_name=figure_name,
        analysis_name="simulation_matplotlib_example",
        save_fig=True,
        save_fig_path=save_fig_path,
        fps=5,
        dpi=100,
    )

    return


def add_default_legend(axes, subplots, traces):
    for subplot in range(subplots):
        legend_items = []
        for trace_dict in traces:
            if subplot in trace_dict.keys():
                for field in trace_dict[subplot]:
                    features = get_features_dictionary(field)
                    legend_items.append(
                        Line2D(
                            [0],
                            [0],
                            color=features["color"],
                            label=features["legend_label"],
                            marker=features["marker"],
                            markersize=3,
                            linestyle=features["linestyle"],
                        )
                    )
        if subplots < 2:
            add_to = axes
        else:
            add_to = axes[subplot]
        add_to.legend(handles=legend_items, loc="upper right")
    return axes


def add_axes(
    traces, num_subplots, axes, data_frames, subplot_titles, time_range=(0, 8)
):
    # Get all of the df[field] for the particular subplot
    # Find the min and max of all of those fields
    # See if contains any of the common glucose bg traces, otherwise do insulin axis
    # Call the function for the appropriate axis
    for subplot in range(num_subplots):
        fields_in_subplot = []
        min_value = 100
        max_value = 0
        for index, traces_per_file in enumerate(traces):
            if subplot in traces_per_file:
                fields = traces_per_file[subplot]
            else:
                fields = []
            fields_in_subplot = fields_in_subplot + fields
            for field in fields:
                if max(data_frames[index][field]) > max_value:
                    max_value = max(data_frames[index][field])
                if min(data_frames[index][field]) < min_value:
                    min_value = min(data_frames[index][field])

        if num_subplots < 2:
            add_to = axes
        else:
            add_to = axes[subplot]

        if subplot < len(subplot_titles):
            add_to.set_title(subplot_titles[subplot])
        if "bg" in fields_in_subplot or "bg_sensor" in fields_in_subplot:
            axes = glucose_axis(
                axes, add_to, time_range, y_range=(min_value, max_value)
            )
        else:
            axes = basal_axis(axes, add_to, time_range, y_range=(0, max_value))
            # set the y-min for the basal axis always to zero
    return axes


def glucose_axis(axes, add_to, time_range=(0, 8), y_range=(70, 250)):
    add_to.set_ylabel("Glucose (mg/dL)")
    add_to.set_xlabel("Hours")
    add_to.set_xlim(time_range[0], time_range[1])
    add_to.set_ylim(y_range[0] - 20, y_range[1] + 50)
    add_to.set_xticks(np.arange(time_range[0], time_range[1] + 1, 2.0))
    add_to.grid(True)
    return axes


def basal_axis(axes, add_to, time_range=(0, 8), y_range=(0, 1)):
    add_to.set_ylabel("Insulin (U or U/hr)")
    add_to.set_xlabel("Hours")
    add_to.set_xlim(time_range[0], time_range[1])
    add_to.set_ylim(y_range[0], y_range[1] + 0.5)
    add_to.set_xticks(np.arange(time_range[0], time_range[1] + 1, 2.0))
    add_to.grid(True)
    return axes


def add_plot(df, num_subplots, axes, field="bg", subplot=2):
    if num_subplots < 2:
        add_to = axes
    else:
        add_to = axes[subplot]

    features_dictionary = get_features_dictionary(field)

    if field in [
        "true_bolus",
        "reported_bolus",
        "true_carb_value",
        "reported_carb_value",
        "true_carb_duration",
        "reported_carb_duration",
        "undelivered_basal_insulin",
    ]:
        # filter out the zero values
        df = df[df[field] != 0]

    if field == "delivered_basal_insulin":
        (markerLines, stemLines, baseLines) = add_to.stem(
            df["hours_post_simulation"],
            df["delivered_basal_insulin"],
            linefmt="#f9706b",
            use_line_collection=True,
        )
        plt.setp(
            markerLines,
            color=features_dictionary["color"],
            markersize=features_dictionary["markersize"],
        )
        plt.setp(
            stemLines,
            color=features_dictionary["color"],
            linewidth=features_dictionary["linewidth"],
        )
        plt.setp(
            baseLines,
            color=features_dictionary["color"],
            linewidth=features_dictionary["linewidth"],
        )
    else:
        add_to.plot(
            df["hours_post_simulation"],
            df[field],
            color=features_dictionary["color"],
            alpha=features_dictionary["alpha"],
            linestyle=features_dictionary["linestyle"],
            linewidth=features_dictionary["linewidth"],
            marker=features_dictionary["marker"],
            markersize=features_dictionary["markersize"],
            drawstyle=features_dictionary["drawstyle"],
        )
    if field == "temp_basal_sbr_if_nan":
        add_to.fill_between(
            df["hours_post_simulation"],
            df[field],
            color=features_dictionary["color"],
            step=features_dictionary["step"],
            alpha=0.4,
        )
    return axes


def generate_figure_from_user_input():
    # Get user input for features in plot
    file_location = os.path.join("..", "..", "data", "processed")

    num_subplots = int(input("How many subplots in the figure?"))

    file_names = list(
        input(
            "What file(s) do you want to visualize in the plot (as comma-separated list)?"
        ).split(",")
    )
    # risk_scenarios_PyLoopkit v0.1.csv

    traces = []

    for file in file_names:
        subplot_dictionary = dict()
        for subplot in range(num_subplots):
            traces_for_subplot = list(
                input(
                    "What fields do you "
                    "want to show from " + str(file) + ""
                    ", in subplot " + str(subplot + 1) + ""
                    " (as comma"
                    " separated list)?"
                ).split(",")
            )
            subplot_dictionary[subplot] = traces_for_subplot
        traces.append(subplot_dictionary)

    # "bg", "bg_sensor"
    # "sbr", "iob", "delivered_basal_insulin"
    # "sbr"

    hours = int(input("How many hours do you want to show?"))  # 8
    title_text = input("Title of plot:")

    create_simulation_figure_matplotlib(
        file_location=file_location,
        file_names=file_names,
        traces=traces,
        subplots=num_subplots,  # type of axis and reference dataframe
        time_range=[0, hours],
        main_title=title_text,
        subplot_titles=[""],
        animate_figure=False,
    )

    return


