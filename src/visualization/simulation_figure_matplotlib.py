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
    save_fig_path=os.path.join(
        "..", "..", "reports", "figures", "simulation_figure_examples"
    ),
):
    """
    Create matplotlib animation or static image of simulation output that generally
    aligns with Tidepool's color scheme/styling for simulation elements (blood glucose,
    scheduled basal rates, temp basals, etc.).

    Parameters
    ----------
    file_location: str
        folder location where files to visualize data from are located
    file_names: list of strings
        list of filenames within that folder that want to pull data from for the animation/figure
    traces: list of dictionaries
        a list of dictionaries where each dictionary corresponds to one of the passed in filenames or dataframes,
        the keys of the dictionaries are subplots (0-indexed), and the values are a list of values for
        that subplot from that filename (ex. traces = [{0: ["bg", "bg_sensor"], 1: ["iob"], 2: ["sbr"]}])
    subplots: int
        number of subplots (1-indexed)
    time_range: tuple
        the range of hours of the simulation want to show (i.e. (0, 8) for
        8 hours starting at t=0
    main_title: str
        main title for the figure
    subplot_titles: list of strings
        list of subplot titles
    animate_figure: bool
        boolean for whether to make the figure an animation or a static figure
    figure_name: str
        the name to use for the figure
    save_fig_path: str
        file path to save figures to

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

    # Initialize Camera for animation
    camera = Camera(fig)

    # Add figure layout elements
    fig.set_size_inches(10, 3 * subplots)
    fig.suptitle(
        main_title, fontsize=10
    )  # fontsize=11, x=.05, horizontalalignment="left")

    # Add axes
    axes = add_axes(
        traces=traces,
        num_subplots=subplots,
        axes=axes,
        data_frames=data_frames,
        time_range=time_range,
        subplot_titles=subplot_titles,
    )

    # Create custom legend and add to axes
    axes = add_default_legend(axes, subplots, traces)

    # Add in different animation traces

    # Determine time values to use for stepping through animation
    if animate_figure:
        time_values = data_frames[0]["hours_post_simulation"]

    # Just use maximum time value in case of no animation
    else:
        time_values = [max(data_frames[0]["hours_post_simulation"])]

    # Iterate over each time value and create a snapshot at that time value
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
    """
    Add legend to the axes of the plot. This is needed to be done using matplotlib shapes
    rather than the build in matplotlib legend because otherwise the animation will add
    a legend at each time step rather than just once.

    Parameters
    ----------
    axes: axes object
        the axes of the matplotlib figure
    subplots: int
        number of subplots in the figure
    traces: list of dictionaries
        a list of dictionaries where each dictionary corresponds to one of the passed in filenames or dataframes,
        the keys of the dictionaries are subplots (0-indexed), and the values are a list of values for
        that subplot from that filename (ex. traces = [{0: ["bg", "bg_sensor"], 1: ["iob"], 2: ["sbr"]}])


    Returns
    -------

    """

    # Add the corresponding shape and label for each field in the plot to the legend
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
        # Syntax is slightly different if there is only 1 subplot
        if subplots < 2:
            add_to = axes
        else:
            add_to = axes[subplot]
        add_to.legend(handles=legend_items, loc="upper right")

    # Return the updated axes

    return axes


def add_axes(
    traces, num_subplots, axes, data_frames, subplot_titles, time_range=(0, 8)
):
    """

    Parameters
    ----------
    traces: list of dictionaries
        a list of dictionaries where each dictionary corresponds to one of the passed in filenames or dataframes,
        the keys of the dictionaries are subplots (0-indexed), and the values are a list of values for
        that subplot from that filename (ex. traces = [{0: ["bg", "bg_sensor"], 1: ["iob"], 2: ["sbr"]}])
    num_subplots: int
        number of subplots (1-indexed)
    axes: axes object
        the axes of the matplotlib figure
    data_frames: list of dataframes
    subplot_titles: list of strings
        list of subplot titles
    time_range: tuple
        the range of hours of the simulation want to show (i.e. (0, 8) for
        8 hours starting at t=0v

    Returns
    -------
    axes: axes object
        the axes of the matplotlib figure updated with the new axes elements

    """
    # Get all of the df[field] for the particular subplot
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

            # Find the min and max of all of the fields in the subplot
            for field in fields:
                if max(data_frames[index][field]) > max_value:
                    max_value = max(data_frames[index][field])
                if min(data_frames[index][field]) < min_value:
                    min_value = min(data_frames[index][field])

        # Get the particular axis going to be updating based on subplot number
        # Different syntax if only one subplot
        if num_subplots < 2:
            add_to = axes
        else:
            add_to = axes[subplot]

        # Set subplot titles
        if subplot < len(subplot_titles):
            add_to.set_title(subplot_titles[subplot])

        # See if subplot contains any of the common glucose bg traces, otherwise do insulin axis
        # Call the function for the appropriate axis
        if "bg" in fields_in_subplot or "bg_sensor" in fields_in_subplot:
            axes = glucose_axis(
                axes, add_to, time_range, y_range=(min_value, max_value)
            )
        else:
            axes = basal_axis(axes, add_to, time_range, y_range=(0, max_value))
            # set the y-min for the basal axis always to zero

    return axes


def glucose_axis(axes, add_to, time_range=(0, 8), y_range=(70, 250)):
    """
    Add elements for a y axis of a plot that's showing glucose data.

    Parameters
    ----------
    axes: axes object
        the axes of the matplotlib figure
    add_to: subplot of figure
        the particular subplot of the figure to use
    time_range: tuple
        the range of hours of the simulation want to show (i.e. (0, 8) for
        8 hours starting at t=0
    y_range: tuple
        range for the y-axis (i.e. (y min, y max) )

    Returns
    -------
    axes: axes object
        the axes of the matplotlib figure updated with the new axis

    """
    # Add elements for a y axis of a plot that's showing glucose data
    add_to.set_ylabel("Glucose (mg/dL)")
    add_to.set_xlabel("Hours")
    add_to.set_xlim(time_range[0], time_range[1])
    add_to.set_ylim(y_range[0] - 20, y_range[1] + 50)
    add_to.set_xticks(np.arange(time_range[0], time_range[1] + 1, 2.0))
    add_to.grid(True)
    return axes


def basal_axis(axes, add_to, time_range=(0, 8), y_range=(0, 1)):
    """
    Add elements for a y axis of a plot that's showing insulin data.

     Parameters
    ----------
    axes: axes object
        the axes of the matplotlib figure
    add_to: subplot of figure
        the particular subplot of the figure to use
    time_range: tuple
        the range of hours of the simulation want to show (i.e. (0, 8) for
        8 hours starting at t=0
    y_range: tuple
        range for the y-axis (i.e. (y min, y max) )

    Returns
    -------
    axes: axes object
        the axes of the matplotlib figure updated with the new axis

    """
    # Add elements for a y axis of a plot that's showing insulin data
    add_to.set_ylabel("Insulin (U or U/hr)")
    add_to.set_xlabel("Hours")
    add_to.set_xlim(time_range[0], time_range[1])
    add_to.set_ylim(y_range[0], y_range[1] + 0.5)
    add_to.set_xticks(np.arange(time_range[0], time_range[1] + 1, 2.0))
    add_to.grid(True)
    return axes


def add_plot(df, num_subplots, axes, field="bg", subplot=2):
    """

    Parameters
    ----------
    df: dataframe
        the dataframe adding data from for the plot
    num_subplots: int
        number of subplots (1-indexed)
    axes: axes object
        the axes of the matplotlib figure
    field: basestring
        the field of the dataframe for the plot
    subplot: int
        the particular subplot number adding this particular trace to

    Returns
    -------
    axes: axes object
        the axes of the matplotlib figure updated with the new plot

    """

    if num_subplots < 2:
        add_to = axes
    else:
        add_to = axes[subplot]

    # Get dictionary of style features for that particular field
    features_dictionary = get_features_dictionary(field)

    # Filter out the zeros for certain fields
    # (i.e. do not want to show a true_bolus of zero)
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

    # For delivered basal insulin, need additional syntax for stem plot
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

    # Add the data trace to the particular subplot with the particular design features
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

    # Add fill in for temp_basal_sbr_if_nan
    if field == "temp_basal_sbr_if_nan":
        add_to.fill_between(
            df["hours_post_simulation"],
            df[field],
            color=features_dictionary["color"],
            step=features_dictionary["step"],
            alpha=0.4,
        )

    # Return updated axes object

    return axes


def generate_figure_from_user_input():
    """
    Generate the matplotlib simulation figure from user input.

    """

    # Get user input for features in plot
    file_location = os.path.join("..", "..", "data", "processed")

    # Get number of subplots from user
    num_subplots = int(input("How many subplots in the figure?"))

    # Get filenames to show in plot from user (ex. risk_scenarios_PyLoopkit v0.1.csv)
    file_names = list(
        input(
            "What file(s) do you want to visualize in the plot (as comma-separated list)?"
        ).split(",")
    )

    traces = []

    # Get fields per filename to show in plot from user
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

    # Get number of hours to show from user
    hours = int(input("How many hours do you want to show?"))  # 8

    # Get title of plot from user
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
