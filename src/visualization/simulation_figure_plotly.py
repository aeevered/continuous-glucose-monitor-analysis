__author__ = "Anne Evered"

# %% REQUIRED LIBRARIES
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import os
from src.visualization.save_view_fig import save_view_fig
from src.visualization.simulation_figures_shared_functions import (
    data_loading_and_preparation,
    get_features_dictionary,
)

# These functions are for creating an animation or static figure of simulation
# output with Plotly that roughly follows Tidepool design color scheme, etc. for
# different simulation elements (bg, temp_basal, sbr, etc.).

# For more examples of animation in plotly please see this reference:
# https://chart-studio.plotly.com/~empet/15243/animating-traces-in-subplotsbr/#/


def add_plot(fig, df, field, row):
    """
    Add a trace to the figure for the particular field and row.

    Parameters
    ----------
    fig: plotly figure object
        the plotly figure object adding plot to
    df: dataframe
        the dataframe the data for the plot comes from
    field: str
        the field of the dataframe to add the plot for
    row: int
        the subplot row to add the plot to (1-indexed)

    Returns
    -------
    fig
        updated Plotly figure object with new subplot

    """

    # Get the features dictionary for the particular field adding to the plot
    features_dictionary = get_features_dictionary(field)

    # Add the trace for the particular field, using the passed in row and
    # the particular style features for that field
    fig.add_trace(
        go.Scatter(
            x=df["hours_post_simulation"],
            y=df[field],
            hoverinfo="y",
            yaxis="y" + str(row),
            line_shape=features_dictionary["shape"],
            line_dash=features_dictionary["dash"],
            mode=features_dictionary["mode"],
            legendgroup=features_dictionary["legend_label"],
            line_color=features_dictionary["color"],
            name=features_dictionary["legend_label"],
            showlegend=True,
            fill=features_dictionary["fill"],
        ),
        row=row,
        col=1,
    )

    # Return updated figure

    return fig


def set_layout(
    traces,
    num_subplots,
    fig,
    data_frames,
    time_range=(0, 8),
    custom_axes_ranges=False,
    custom_tick_marks=False,
):
    """

    Parameters
    ----------
    traces: list of dictionaries
        a list of dictionaries where each dictionary corresponds to one of the passed in filenames,
        the keys of the dictionaries are subplots (0-indexed), and the values are a list of values for
        that subplot from that filename (ex. traces = [{0: ["bg", "bg_sensor"], 1: ["iob"], 2: ["sbr"]}])
    num_subplots: int
        the total number of subplots in the figure
    fig: figure object
        the plotly figure object
    data_frames: list
        the list of dataframes corresponding to the files
    time_range: int
        the range of hours of the simulation want to show (i.e. (0, 8) for
        8 hours starting at t=0
    custom_axes_ranges: list of ranges (length of number of subplots)
        custom axis ranges to use if do not want to use the ones specified
        based on the min and max values of the fields for that subplot
    custom_tick_marks:  list of lists (length of number of subplots)
        custom axis ranges to use if do not want to use the ones specified
        based on the min and max values of the fields for that subplot

    Returns
    -------
    fig
        plotly figure with layout specified

    """
    # Optional setting layout width and height
    # fig.update_layout(width=700, height=475)

    # Set the x axis range based on the passed in time ranges
    fig.update_xaxes(range=[time_range[0], time_range[1]], title="Hours")

    # Set the layout for each of the subplots
    for subplot in range(num_subplots):
        # Get min and max value
        fields_in_subplot = []

        # Set initial min and max value as placeholders
        min_value = 100
        max_value = 0

        # Get all of the fields in that subplot: Iterate over all of the dictionaries in traces and
        # if there are fields for the particular subplot in that dictionary, add them to fields_in_subplot.
        # Update the min and max values for the subplot so can set the axis ranges to the overall
        # min and max of all fields in that subplot.
        for index, traces_per_file in enumerate(traces):
            if subplot in traces_per_file:
                fields = traces_per_file[subplot]
            else:
                fields = []
            fields_in_subplot = fields_in_subplot + fields
            for field in fields:
                if np.nanmax(data_frames[index][field]) > max_value:
                    max_value = np.nanmax(data_frames[index][field])
                if np.nanmin(data_frames[index][field]) < min_value:
                    min_value = np.nanmin(data_frames[index][field])

        # If bg is in that fields in a given subplot, update the y axis
        # with the min and max found above the a title of "Glucose (mg/dL)"
        if "bg" in fields_in_subplot or "bg_sensor" in fields_in_subplot:
            fig.update_yaxes(
                range=[min_value - 20, max_value + 50],
                title="Glucose (mg/dL)",
                row=subplot + 1,
                col=1,
            )
        # Otherwise set to an "insulin axis" using 0 as the minimum and
        # "Insulin (U/hr) as the axis title
        else:
            fig.update_yaxes(
                range=[0, max_value + 0.5],
                title="Insulin (U/hr)",
                row=subplot + 1,
                col=1,
            )

    # If there were custom axes ranges specified, update the y axes again for those ranges
    if custom_axes_ranges is not False:
        for subplot in range(num_subplots):
            fig.update_yaxes(
                range=[custom_axes_ranges[subplot][0], custom_axes_ranges[subplot][1]],
                row=subplot + 1,
                col=1,
            )

    # If there were custome tick marks specified, update the y axes for those tick marks
    if custom_tick_marks is not False:
        for subplot in range(num_subplots):
            fig.update_yaxes(
                tickmode="array",
                tickvals=custom_tick_marks[subplot],
                row=subplot + 1,
                col=1,
            )

    # Return the figure with the updated layout.

    return fig


# Create figure
def create_simulation_figure_plotly(
    file_location,
    file_names,
    traces,
    subplots,
    data_frames=[],
    files_need_loaded=False,
    show_legend=True,
    time_range=(0, 8),
    main_title="Risk Scenario Simulation",
    subplot_titles=[],
    save_fig_path="",
    subtitle="",
    figure_name="simulation_figure",
    analysis_name="risk-scenarios",
    animate=True,
    custom_axes_ranges=False,
    custom_tick_marks=False,
):
    """
    Create plotly animation or static image of simulation output that generally
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
    data_frames: list of dataframes
        list of dataframes to show data from in the animation/figure
        can leave as a blank list (default) and pass in file_names instead and files_need_loaded as True
    files_need_loaded: bool
        boolean for whether the files need loaded from file location/name or whether you are passing in data_frames
    show_legend: bool
        boolean for whether or not want to show the legend on figure
    time_range: tuple
        the range of hours of the simulation want to show (i.e. (0, 8) for
        8 hours starting at t=0
    main_title: str
        main title for the figure
    subplot_titles: list of strings
        list of subplot titles
    save_fig_path: str
        file path to save figures to
    subtitle: str
        subtitle to add to figure
    figure_name: str
        the name to use for the figure
    analysis_name: str
        the name of the analysis this figure is a part of
    animate: bool
        boolean for whether to make the figure an animation or a static figure
    custom_axes_ranges: list of tuples (length of list should be equal to number of subplots)
        custom axis ranges to use if do not want to use the ones specified
        based on the min and max values of the fields for that subplot
    custom_tick_marks:  list of lists (length of list should be equal to number of subplots)
        custom axis ranges to use if do not want to use the ones specified
        based on the min and max values of the fields for that subplot

    Returns
    -------

    """
    # Load data files
    if files_need_loaded:
        data_frames = []
        for file in file_names:
            df = data_loading_and_preparation(
                os.path.abspath(os.path.join(file_location, file))
            )
            data_frames.append(df)

    # Set up figure and axes
    fig = make_subplots(rows=subplots, cols=1, subplot_titles=np.array(subplot_titles))

    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=14)

    fig = set_layout(
        traces,
        subplots,
        fig,
        data_frames,
        time_range,
        custom_axes_ranges=custom_axes_ranges,
        custom_tick_marks=custom_tick_marks,
    )

    # Add plots
    for index, df in enumerate(data_frames):
        for subplot in traces[index]:
            for field in traces[index][subplot]:
                row = subplot + 1
                fig = add_plot(fig, df, field, row)

    # add annotations
    fig["layout"]["annotations"] += (
        dict(
            xref="paper",
            yref="paper",
            x=0.55,
            y=-0.27,
            showarrow=False,
            text="This visual shows the results of running the simulation for "
            + str(time_range[1])
            + " hours.",
            font=dict(size=13),
        ),
        dict(
            xref="paper",
            yref="paper",
            x=0,
            y=1.05,
            showarrow=False,
            text=subtitle,
            font=dict(size=12),
        ),
    )

    # If animate is true, add in the animation elements
    if animate:

        # For a basic example of animation in plotly please see this reference:
        # https://chart-studio.plotly.com/~empet/15243/animating-traces-in-subplotsbr/#/
        # That example was used as the model for the code in this section.

        # TODO: add in functionality to be able to specify step size and speed of animation as function parameter

        # Specify the number of steps (here specified as the steps being every 15 minutes
        # from start time to end time)
        time_chunks = list(np.arange(time_range[0], time_range[1] + 0.2, 0.25))
        num_frames = len(time_chunks)

        frames = []

        # For each step (where number of steps is specified by num_frames)
        for k in range(num_frames):
            data = []
            num_traces = 0
            for index, df in enumerate(data_frames):
                # Animate each field in each subplot
                for subplot in traces[index]:
                    for field in traces[index][subplot]:
                        # Extend the data for that step
                        data.extend(
                            [
                                go.Scatter(
                                    y=df[df["hours_post_simulation"] <= time_chunks[k]][
                                        field
                                    ]
                                )
                            ]
                        )
                        num_traces += 1
            frames.append(
                dict(
                    name=k,
                    data=data,
                    # update the trace 1 in (1,1)
                    traces=list(np.arange(num_traces))
                    # [0, 1, 2]  # the elements of the list [0,1,2] give info on the traces in fig.data
                    # that are updated by the above three go.Scatter instances
                )
            )

        # Add update menus (play and pause) to animation
        updatemenus = [
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label=">",
                        method="animate",
                        args=[
                            [f"{k}" for k in range(num_frames)],
                            dict(
                                frame=dict(duration=500, redraw=False),
                                transition=dict(duration=0),
                                easing="linear",
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="=",
                        method="animate",
                        args=[
                            [],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
                direction="left",
                pad=dict(r=10, t=85),
                showactive=False,
                x=0.08,
                y=-0.05,
                xanchor="right",
                yanchor="top",
            )
        ]

        # Add in the slider so can move to different times after the animation is completed or paused.
        sliders = [
            {
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Frame: ",
                    "visible": False,
                    "xanchor": "right",
                },
                "transition": {"duration": 500.0, "easing": "linear"},
                "pad": {"b": 10, "t": 50},
                "len": 1,
                "x": 0,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [k],
                            {
                                "frame": {
                                    "duration": 500.0,
                                    "easing": "linear",
                                    "redraw": False,
                                },
                                "transition": {"duration": 0, "easing": "linear"},
                            },
                        ],
                        "label": time_chunks[k],
                        "method": "animate",
                    }
                    for k in range(num_frames)
                ],
            }
        ]

        # This step is very important - the animation will not work if this is not specified.
        fig.update(frames=frames),

        # Add title, updatemenus, and sliders to figure layout
        fig.update_layout(
            title_text=main_title,
            title_font_size=16,
            margin_b=150,
            margin_t=90,
            updatemenus=updatemenus,
            sliders=sliders,
            font=dict(size=10),
        )

        # Optional controls of animation speed
        # fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200

    # If animation is not True, just update the layout for a static figure.
    else:
        fig.update_layout(
            title_text=main_title,
            title_font_size=16,
            margin_b=150,
            margin_t=90,
            font=dict(size=10),
        )

    # If show_legend if false, update the layout to not have a legend.
    if show_legend is False:
        fig.update_layout(showlegend=False)

    # Save and/or view figure
    save_view_fig(
        fig,
        image_type="png",
        figure_name=figure_name,
        analysis_name=analysis_name,
        view_fig=True,
        save_fig=True,
        save_fig_path=save_fig_path,
        width=1200,
        height=700,
    )

    return
