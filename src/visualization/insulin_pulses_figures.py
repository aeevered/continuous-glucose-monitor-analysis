__author__ = "Anne Evered"

# %% REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from src.visualization.save_view_fig import save_view_fig

# Define color dictionaries
risk_score_dict = {
    0: "0 - None",
    1: "1 - Negligible",
    2: "2 - Minor",
    3: "3 - Serious",
    4: "4 - Critical",
}
risk_score_color_dict = {
    "0 - None": "#0F73C6",
    "1 - Negligible": "#06B406",
    "2 - Minor": "#D0C07F",
    "3 - Serious": "#E18325",
    "4 - Critical": "#9A3A39",
}


# Visualization Functions
def create_scatter_plot_v1(
        table_df,
        x_value,
        y_value,
        hover_value,
        color_value_column,
        legend_title,
        title,
        x_title,
        y_title,
        color_dict=risk_score_color_dict,
        score_dict=risk_score_dict,
        image_type="png",
        figure_name="<number-or-name>",
        analysis_name="<analysis-name>",
        view_fig=True,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
        width=600,
        height=700,
):
    """
    Version one (in terms of design) of scattterplot showing breakdown
    of DKAI risk scores.

    Parameters
    ----------
    table_df: dataframe
        dataframe of summary results to display in scatterplot
    x_value: str
        column name of table_df to use for x in plot
    y_value: str
        column name of table_df to use for x in plot
    hover_value: str
        column name of table_df to use for hover in plot
    color_value_column: str
        column name of table_df to use for color value in plot
    color_dict: dictionary
        dictionary mapping risk score strings (ex. "0-None", "1-Negligible","2-Minor", etc.)
        to color values for representing that risk category (ex. "4 - Critical": "red")
    score_dict: dictionary
        dictionary mapping risk score ints (ex. 0-4) to strings for that risk
        score (ex. "0-None", "1-Negligible","2-Minor", etc.)
    legend_title: str
        title for legend
    title: str
        title for plot
    x_title: str
        title for x axis
    y_title: str
        title for y axis
    image_type: str
        file type ("png","jpg","pdf", etc.) to save image as
    figure_name: str
        name for figure
    analysis_name: str
        name of analysis this figure falls under (used in name for saving the figure)
    view_fig: bool
        whether or not you want to view the figure (opens in browser)
    save_fig: bool
        whether or not you want to save the figure
    save_fig_path: str
        file location to save the figure at
    width: int
        desired width of the saved figure in pixels
    height: int
        desired height of the saved figure in pixels

    Returns
    -------

    """
    traces = []

    # For all of the risk categories, append scatterpoints corresponding to
    # the values in that risk category
    for value in sorted(table_df[color_value_column].unique()):
        df = table_df[table_df[color_value_column] == value]

        traces.append(
            go.Scattergl(
                x=df[x_value],
                y=df[y_value],
                mode="markers",
                hovertext=table_df[hover_value],
                name=score_dict[value],
                showlegend=True,
                marker=dict(
                    color=[color_dict[score_dict[value]]] * len(df.index),
                    size=8,
                    line_width=1,
                    opacity=0.7,
                ),
            )
        )

    # Set layout elements
    layout = go.Layout(
        title=title,
        legend_title_text=legend_title,
        xaxis=dict(
            title=x_title,
            gridcolor="rgb(255, 255, 255)",
            gridwidth=2,
            tickmode="linear",
            tick0=0,
            dtick=0.05,
        ),
        yaxis=dict(
            title=y_title,
            gridcolor="rgb(255, 255, 255)",
            gridwidth=2,
            type="linear",
            tick0=0,
            dtick=0.5,
        ),
        paper_bgcolor="rgb(243, 243, 243)",
        plot_bgcolor="rgb(243, 243, 243)",
    )

    # Create figure
    fig = go.Figure(data=traces, layout=layout)

    # Save and/or view figure
    save_view_fig(
        fig=fig,
        image_type=image_type,
        figure_name=figure_name,
        analysis_name=analysis_name,
        view_fig=view_fig,
        save_fig=save_fig,
        save_fig_path=save_fig_path,
        width=width,
        height=height,
    )

    return


def create_scatter_plot_v2(
        table_df,
        color_dict=risk_score_color_dict,
        image_type="png",
        figure_name="<number-or-name>",
        analysis_name="<analysis-name>",
        view_fig=True,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
        width=600,
        height=700,
):
    """

    Parameters
    ----------
    table_df: dataframe
        dataframe of summary results to display in scatterplot
    color_dict: dictionary
        dictionary mapping risk score strings (ex. "0-None", "1-Negligible","2-Minor", etc.)
        to color values for representing that risk category (ex. "4 - Critical": "red")
    image_type: str
        file type ("png","jpg","pdf", etc.) to save image as
    figure_name: str
        name for figure
    analysis_name: str
        name of analysis this figure falls under (used in name for saving the figure)
    view_fig: bool
        whether or not you want to view the figure (opens in browser)
    save_fig: bool
        whether or not you want to save the figure
    save_fig_path: str
        file location to save the figure at
    width: int
        desired width of the saved figure in pixels
    height: int
        desired height of the saved figure in pixels

    Returns
    -------

    """

    # Create scatter plot
    summary_fig = px.scatter(
        data_frame=table_df,
        x="sbr",
        y="loop_max_basal_rate",
        log_y=True,
        color="dka_risk_score_str",
        size="dka_index_circle_size",
        color_discrete_map=color_dict,
        range_y=[0.05, 7.5],
    )

    # Create layout
    layout = go.Layout(
        title="Risk of DKA Associated with Missing Insulin Pulses",
        showlegend=True,
        yaxis=dict(
            title="Maximum Allowable Basal Rate (Loop Setting)",
            tickvals=[0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 5, 7],
        ),
        xaxis=dict(
            title="Scheduled Basal Rate (U/hr)", tickvals=np.arange(0.05, 0.75, 0.05)
        ),
        plot_bgcolor="#D3D3D3",
        legend_title="Tidepool DKAI Risk Score",
    )

    # Add layout
    summary_fig.update_layout(layout)

    # Save and/or view figure
    save_view_fig(
        fig=summary_fig,
        image_type=image_type,
        figure_name=figure_name,
        analysis_name=analysis_name,
        view_fig=view_fig,
        save_fig=save_fig,
        save_fig_path=save_fig_path,
        width=width,
        height=height,
    )

    return


# Simulation Example Plot
def make_scatter_trace(
        x_value, y_value, color, symbol, name, style, dash, line_shape, fill, opacity, size
):
    """

    Creates a scatter trace object based on specified data fields and style features
    and returns that trace object to be added to a plot.

    Parameters
    ----------
    x_value: series
        series of values to plot for x
    y_value: series
        series of values to plot for y
    color: str
        color to use for scatter trace
    symbol: str
        what symbol type to use for scatter trace
    name: str
        name of the plot
    style: str
        style of the plot (i.e. "lines", "markers")
    dash: str
        dash style of the lines (i.e. "dot", "solid)
    line_shape: str
        line shape for the plot (i.e. "hv")
    fill: bool
        whether or not the line should be filled underneath
    opacity: float
        opacity of lines
    size: float
        size of the markers

    Returns
    -------
    trace: trace object
        trace object that can be added to a plot

    """

    # Create scatter trace object
    trace = go.Scatter(
        x=x_value,
        y=y_value,
        hoverinfo="y+name+x",
        name=name,
        mode=style,
        fill=fill,
        opacity=opacity,
        line=dict(shape=line_shape, color=color, dash=dash),
        marker=dict(size=size, line=dict(width=0), color=color, symbol=symbol),
    )

    # Return trace object

    return trace


def make_bar_trace(
        sim_df, x_value, y_value, marker_color, name,
):
    """

    Parameters
    ----------
    sim_df: dataframe
        dataframe to use for plot data
    x_value: str
        column name to use from dataframe for x value plot data
    y_value: str
        column name to use from dataframe for y value plot data
    marker_color: str
        color for markers
    name: str
        name for plot

    Returns
    -------
    trace: trace object
        trace object that can be added to a plot


    """

    # Create bar plot trace object
    trace = go.Bar(
        x=sim_df[x_value],
        y=sim_df[y_value],
        name=name,
        marker_color=marker_color,
        showlegend=False,
    )

    # Return trace object

    return trace


def x_axis_properties(sim_df, show_title_axis_marks):
    """
    Get x axis properties for plot

    Parameters
    ----------
    sim_df: dataframe
        dataframe pulling data from for plot (use for determining x-axis range)
        based on time range in dataframe
    show_title_axis_marks: bool
        whether or not tick labels should be shown in the plot

    Returns
    -------
    x_axis_properties_dict: dictionary
        dictionary of x axis properties that can be applied to plot

    """

    # Set axis title based on parameter
    if show_title_axis_marks:
        title_text = "Hours Post Simulation"
    else:
        title_text = ""

    # Create dictionary of x axis properties
    x_axis_properties_dict = dict(
        range=(0, max(sim_df["hours_post_simulation"])),
        showgrid=True,
        gridcolor="rgb(255, 255, 255)",
        hoverformat="%H:%M",
        showticklabels=show_title_axis_marks,
        tickmode="linear",
        dtick=1,
        tickfont=dict(size=10),
        title=dict(text=title_text, font=dict(size=10)),
    )

    # Return x axis properties dictionary
    return x_axis_properties_dict


def y_axis_properties(title_text):
    """

    Parameters
    ----------
    title_text: str
        text to use for title

    Returns
    -------
    y_axis_properties_dict: dictionary
        dictionary of y axis properties that can be applied to plot

    """

    # Create dictionary of y axis properties
    y_axis_properties_dict = dict(
        showgrid=True,
        gridcolor="rgb(255, 255, 255)",
        hoverformat="%H:%M",
        tickfont=dict(size=10),
        title=dict(text=title_text, font=dict(size=10)),
    )

    # Return y axis properties dictionary
    return y_axis_properties_dict


# TODO: Replace with generic simulation_figure_matplotlib.py or simulation_figure_plotly.py

# Note: this create_simulation_figure code predates the more generalized functions
# for viewing/animating the results of a simulation (simulation_figure_matplotlib.py,
# simulation_figure_plotly.py). These functions could be replaced by those more
# generalized versions at some point.


def create_simulation_figure(
        sim_df,
        image_type="png",
        figure_name="<number-or-name>",
        analysis_name="<analysis-name>",
        view_fig=True,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
        width=600,
        height=900,
):
    """

    Parameters
    ----------
    sim_df: dataframe
        simulation df to visualize
    image_type: str
        file type ("png","jpg","pdf", etc.) to save image as
    figure_name: str
        name for figure
    analysis_name: str
        name of analysis this figure falls under (used in name for saving the figure)
    view_fig: bool
        whether or not you want to view the figure (opens in browser)
    save_fig: bool
        whether or not you want to save the figure
    save_fig_path: str
        file location to save the figure at
    width: int
        desired width of the saved figure in pixels
    height: int
        desired height of the saved figure in pixels

    Returns
    -------

    """

    # Add additional needed fields to simulation dataframe
    sim_df["five_minute_marks"] = sim_df.index
    sim_df["minutes_post_simulation"] = sim_df["five_minute_marks"].apply(
        lambda x: x * 5
    )
    sim_df["hours_post_simulation"] = sim_df["minutes_post_simulation"] / 60

    # Make Subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "BG Over Time",
            "Delivered Basal Insulin",
            "Undelivered Basal Insulin Pulses",
        ),
        vertical_spacing=0.15,
    )

    # Define all fields using and style elements
    y_fields = [
        "bg",
        "bg_sensor",
        "sbr",
        "delivered_basal_insulin",
        "undelivered_basal_insulin",
    ]
    colors = [
        "#472A87",
        "#D9CEED",
        "#f95f3b",
        "#008ECC",
        "#9A3A39",
    ]  # "#008ECC", "#f1c232"]
    names = [
        "true_bg",
        "sensor_bg",
        "sbr",
        "delivered_basal_insulin",
        "undelivered_basal_insulin",
    ]
    symbols = ["circle", "circle", "circle", "circle", "circle"]
    # For stem plots
    styles = ["markers", "markers", "lines", "lines", "markers"]
    sizes = [4, 4, 4, 3, 6]
    dashes = ["solid", "solid", "dot", "solid", "solid"]
    line_shapes = ["hv", "hv", "hv", "hv", "hv"]
    rows = [1, 1, 2, 2, 3]
    fills = [None, None, None, None, None]
    opacities = [0.9, 0.75, 1, 1, 1]

    y_axis_labels = ["BG (mg/dL)", "Insulin (U or U/hr)", "Insulin (U)", "Insulin(U)"]

    fig.append_trace(
        make_bar_trace(
            sim_df=sim_df,
            x_value="hours_post_simulation",
            y_value="undelivered_basal_insulin",
            marker_color="#9A3A39",
            name="Undelivered Basal Insulin",
        ),
        row=3,
        col=1,
    )

    # Make and add all the traces based on style elements
    for (
            y_field,
            color,
            name,
            symbol,
            row,
            style,
            dash,
            line_shape,
            fill,
            opacity,
            size,
    ) in zip(
        y_fields,
        colors,
        names,
        symbols,
        rows,
        styles,
        dashes,
        line_shapes,
        fills,
        opacities,
        sizes,
    ):
        trace = make_scatter_trace(
            x_value=sim_df["hours_post_simulation"],
            y_value=sim_df[y_field],
            color=color,
            symbol=symbol,
            name=name,
            style=style,
            dash=dash,
            line_shape=line_shape,
            opacity=opacity,
            fill=fill,
            size=size,
        )
        fig.append_trace(trace, row=row, col=1)

    # TODO: This is only updating for one of the traces
    fig.update_layout(showlegend=True, paper_bgcolor="white", plot_bgcolor="#D3D3D3")

    # Update x axis properties
    for i in range(1, 5):
        fig.update_xaxes(
            x_axis_properties(sim_df, show_title_axis_marks=True), row=i, col=1
        )
        fig.update_yaxes(y_axis_properties(y_axis_labels[i - 1]), row=i, col=1)

    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=12)

    # Save and view figure
    save_view_fig(
        fig=fig,
        image_type=image_type,
        figure_name=figure_name,
        analysis_name=analysis_name,
        view_fig=view_fig,
        save_fig=save_fig,
        save_fig_path=save_fig_path,
        width=width,
        height=height,
    )

    return


if __name__ == '__main__':

    # Read in data file
    insulin_pulse_file_location = os.path.join("..", "..", "data", "raw", "insulin-pulses-sample-files-2020-07-02")
    summary_metrics_file = "2020-07-02-06-41-20-summary.csv"

    summary_metrics_path = os.path.abspath(
        os.path.join(insulin_pulse_file_location, summary_metrics_file)
    )

    summary_metrics_df = pd.read_csv(summary_metrics_path)

    # Sort so that legend is in the right order
    summary_metrics_df.sort_values(
        "dka_risk_score", inplace=True
    )

    # Add circle size for dka risk index points
    summary_metrics_df["dka_index_circle_size"] = summary_metrics_df["dka_index"] + 1.0

    # Add string for risk scores for legend from dictionary
    summary_metrics_df["dka_risk_score_str"] = summary_metrics_df["dka_risk_score"].replace(
        risk_score_dict
    )

    # Create scatter plot figures (both version 1 design and version 2 design)
    create_scatter_plot_v1(table_df=summary_metrics_df, x_value="sbr", y_value="loop_max_basal_rate",
                           hover_value="dka_index", color_value_column="dka_risk_score", legend_title="DKAI Risk Score",
                           title="DKAI Risk Score by Basal Rate", x_title="Scheduled Basal Rate",
                           y_title="Loop Max Allowable Basal Rate", color_dict=risk_score_color_dict,
                           score_dict=risk_score_dict,
                           image_type="png", figure_name="summary-metrics-dkai-riskscore-scatterplot-v1",
                           analysis_name="insulin-pulses", view_fig=True, save_fig=True, save_fig_path=os.path.join(
            "..",
            "..",
            "reports",
            "figures",
            "insulin-pulses-risk-assessment"
        ), width=600, height=700)

    create_scatter_plot_v2(
        table_df=summary_metrics_df,
        color_dict=risk_score_color_dict,
        image_type="png",
        figure_name="summary-metrics-dkai-riskscore-scatterplot-v2",
        analysis_name="insulin-pulses",
        view_fig=True,
        save_fig=False,
        save_fig_path=os.path.join(
            "..",
            "..",
            "reports",
            "figures",
            "insulin-pulses-risk-assessment"
        ),
        width=600,
        height=500,
    )

    # Create insulin pulse simulation example figure

    # Iterate through all of the files
    for filename in os.listdir(insulin_pulse_file_location):

        # Only create figures for .csv files and not for the summary file
        if filename.endswith(".csv") and not ("summary.csv" in filename):
            print(filename)

            # Read in the simulation file data
            simulation_example_path = os.path.abspath(
                os.path.join(insulin_pulse_file_location, filename)
            )

            simulation_example_df = pd.read_csv(simulation_example_path)

            # Create the simulation figure
            create_simulation_figure(
                sim_df=simulation_example_df,
                image_type="png",
                figure_name="simulation-visualization-" + filename,
                analysis_name="insulin-pulses",
                view_fig=False,
                save_fig=True,
                save_fig_path=os.path.join(
                    "..",
                    "..",
                    "reports",
                    "figures",
                    "insulin-pulses-risk-assessment"
                ),
                width=800,
                height=1000,
            )

            continue
        else:
            continue
