# %% REQUIRED LIBRARIES
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio
import datetime as dt

utc_string = dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%m-%S")
code_version = "v0-1-0"

insulin_pulse_file_location = os.path.join("..", "..", "data", "raw", "insulin-pulses-risk-assessment")
summary_metrics_file = "2020-06-24-insulin-pulse-summary-stats-nogit.csv"
simulation_example_file = "SBR 0.05 VPBR 0.05 MBR 0.1-nogit.csv"

summary_metrics_path = os.path.abspath(os.path.join(insulin_pulse_file_location, summary_metrics_file))
simulation_example_path = os.path.abspath(os.path.join(insulin_pulse_file_location, simulation_example_file))

summary_metrics_df = pd.read_csv(summary_metrics_path)
simulation_example_df = pd.read_csv(simulation_example_path)


## Summary Plot
# Just need to plot the DKA Risk Score
# Use the 5 colors sketched out in this doc:
# https://docs.google.com/document/d/1XBUKXTzhWR49ZS63mDXlkTePFK1KDmpRZsPtz3_J06E/edit?ts=5ef2527a#
# Cameron did a heatmap, but I think we can do a plotly scattergl plot.
# BONUS: add the DKA index value to the hover state


# from make_figures_and_tables import save_view_fig

def save_view_fig(
        fig,
        image_type="png",
        figure_name="<number-or-name>",
        analysis_name="analysis-<name>",
        view_fig=True,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
        width=600,
        height=700
):
    if view_fig:
        fig.show()

    file_name = "{}-{}_{}_{}".format(analysis_name, figure_name, utc_string, code_version)

    if save_fig:
        pio.write_image(
            fig=fig, file=os.path.join(save_fig_path, file_name + ".{}".format(image_type)), format=image_type,
            width=width, height=height

        )

    return


def create_scatterplot(
        table_df,
        x_value,
        y_value,
        hover_value,
        color_value_column,
        color_dict,
        score_dict,
        legend_title,
        title,
        x_title,
        y_title,
        image_type="png",
        figure_name="<number-or-name>",
        analysis_name="<analysis-name>",
        view_fig=True,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
        width=600,
        height=700
):
    traces = []

    for value in sorted(table_df[color_value_column].unique()):
        df = table_df[table_df[color_value_column] == value]

        traces.append(go.Scattergl(
            x=df[x_value],
            y=df[y_value],
            mode='markers',
            hovertext=table_df[hover_value],
            name=str(value) + " - " + score_dict[value],
            showlegend=True,
            marker=dict(
                color=[color_dict[value]] * len(df.index),
                size=8,
                line_width=1,
                opacity=0.7
            )
        ))

    layout = go.Layout(
        title=title,
        legend_title_text=legend_title,
        xaxis=dict(
            title=x_title,
            gridcolor='rgb(255, 255, 255)',
            gridwidth=2,
            tickmode='linear',
            tick0=0,
            dtick=0.05
        ),
        yaxis=dict(
            title=y_title,
            gridcolor='rgb(255, 255, 255)',
            gridwidth=2,
            type='linear',
            tick0=0,
            dtick=0.5
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)')

    fig = go.Figure(data=traces, layout=layout)

    save_view_fig(fig=fig
                  , image_type=image_type
                  , figure_name=figure_name
                  , analysis_name=analysis_name
                  , view_fig=view_fig
                  , save_fig=save_fig
                  , save_fig_path=save_fig_path
                  , width=width
                  , height=height)

    return


# Create Scatterplot Figure

# Define color dictionary
color_dict = {0: "#0F73C6",
              1: "#06B406",
              2: "#D0C07F",
              3: "#E18325",
              4: "#9A3A39"}

score_dict = {0: "None",
              1: "Negligible",
              2: "Minor",
              3: "Serious",
              4: "Critical"}

create_scatterplot(
    table_df=summary_metrics_df,
    x_value="sbr",
    y_value="loop_max_basal_rate",
    hover_value="dka_index",
    color_value_column="dka_risk_score",
    color_dict=color_dict,
    score_dict=score_dict,
    legend_title="DKAI Risk Score",
    title="DKAI Risk Score by Basal Rate",
    x_title="Scheduled Basal Rate",
    y_title="Loop Max Allowable Basal Rate",
    image_type="png",
    figure_name="summary-metrics-dkai-riskscore-scatterplot",
    analysis_name="insulin-pulses",
    view_fig=True,
    save_fig=True,
    save_fig_path=os.path.join("..", "..", "reports", "figures", "insulin-pulses-risk-assessment"),
    width=600,
    height=700

)

# Simulation Example Plot
# See here for example figures:
# https://colab.research.google.com/drive/1oVWDc734_RndivI5lAcc8dqxtwft5kMA


## Prepare data

##For BG traces

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def make_scatter_trace(
        x_value,
        y_value,
        color,
        symbol,
        name,
        style,
        dash,
        line_shape,
        fill,
        opacity,
        size

):
    trace = go.Scatter(
        x=x_value,
        y=y_value,
        hoverinfo="y+name+x",
        name=name,
        mode=style,
        fill=fill,
        opacity=opacity,
        line=dict(
            shape=line_shape,
            color=color,
            dash=dash
        ),
        marker=dict(
            size=size,
            line=dict(width=0),
            color=color,
            symbol=symbol
        )
    )

    return trace

def make_bar_trace(
        sim_df,
        x_value,
        y_value,
        marker_color,
        name,

):
    trace = go.Bar(x=sim_df[x_value]
                , y=sim_df[y_value]
                , name=name
                , marker_color=marker_color
                , showlegend=False)

    return trace


def x_axis_properties(sim_df, show_title_axis_marks):
    if show_title_axis_marks:
        title_text = "Minutes Post Simulation"
    else:
        title_text = ""

    x_axis_properties = dict(
        range=(0, max(sim_df["minutes_post_simulation"])),
        showgrid=True,
        gridcolor="rgb(255, 255, 255)",
        hoverformat="%H:%M",
        showticklabels=show_title_axis_marks,
        tickmode='linear',
        dtick=60,
        tickfont=dict(
            size=10
        ),
        title=dict(
            text=title_text,
            font=dict(
                size=10
            )
        )
    )

    return x_axis_properties


def y_axis_properties(title_text):
    y_axis_properties = dict(
        showgrid=True,
        gridcolor="rgb(255, 255, 255)",
        hoverformat="%H:%M",
        tickfont=dict(
            size=10
        ),
        title=dict(
            text=title_text,
            font=dict(
                size=10
            )
        )
    )

    return y_axis_properties


def create_simulation_figure(sim_df,
                             image_type="png",
                             figure_name="<number-or-name>",
                             analysis_name="<analysis-name>",
                             view_fig=True,
                             save_fig=True,
                             save_fig_path=os.path.join("..", "..", "reports", "figures"),
                             width=600,
                             height=900
                             ):

    sim_df['five_minute_marks'] = sim_df.index
    sim_df['minutes_post_simulation'] = sim_df['five_minute_marks'].apply(lambda x: x * 5)

    #Make Subplots
    fig = make_subplots(rows=4
                        , cols=1
                        , subplot_titles=("BG Over Time", "Insulin Delivery", "Delivered Basal Insulin", "Undelivered Basal Insulin")
                        , vertical_spacing=0.15)

    #Define all fields using and style elements
    y_fields = ["bg", "bg_sensor", "sbr", 'temp_basal', 'bolus', 'delivered_basal_insulin', 'undelivered_basal_insulin']
    colors = ["#472A87", "#D9CEED", "#008ECC", "#008ECC", "#6AA84F", "#f95f3b", "#f95f3b"] # "#008ECC", "#f1c232"]
    names = ["true_bg", "sensor_bg",  "sbr", 'temp_basal', 'bolus', 'delivered_basal_insulin', 'undelivered_basal_insulin']
    symbols = ['circle', "circle",'circle', 'circle', 'triangle-down', 'circle', 'circle']

    #For stem plots
    styles = ['markers', "markers",'lines', 'lines', 'markers', 'markers', 'markers']
    sizes = [4, 4, 4, 4, 4, 3, 3]

    #To make not stem plots, uncomment following line
    #styles = ['markers', "markers", 'lines', 'lines', 'markers', 'lines', 'lines']
    #sizes = [4, 4, 4, 4, 4, 4, 4]

    dashes = ["solid", 'solid', "dot", "solid", "solid", 'solid', 'solid']
    line_shapes = ['hv', 'hv', 'hv', 'hv', 'hv', 'hv', 'hv']
    rows = [1, 1, 2, 2, 2, 3, 4]
    fills = [None, None, None, None, 'tozeroy', None, None]
    opacities = [.9, .75, 1, 1, 0.25, 1, 1]

    y_axis_labels = ["BG (mg/dL)", "Insulin (U or U/hr)", "Insulin (U)", "Insulin(U)"]

    # Make and add all the traces
    for y_field, color, name, symbol, row, style, dash, line_shape, fill, opacity, size in zip(y_fields, colors, names,
                                                                                         symbols, rows, styles, dashes,
                                                                                         line_shapes, fills, opacities, sizes):
        trace = make_scatter_trace(
            x_value=sim_df["minutes_post_simulation"],
            y_value=sim_df[y_field],
            color=color,
            symbol=symbol,
            name=name,
            style=style,
            dash=dash,
            line_shape=line_shape,
            opacity=opacity,
            fill=fill,
            size=size

        )
        fig.append_trace(trace, row=row, col=1)



    #Add in the barplot traces to make these stem plots (REMOVE FOR NO STEM PLOT)
    fig.append_trace(make_bar_trace(
        sim_df=sim_df,
        x_value="minutes_post_simulation",
        y_value="delivered_basal_insulin",
        marker_color= "#008ECC",
        name="Delivered Basal Insulin",
    ), row=3, col=1)

    fig.append_trace(make_bar_trace(
        sim_df=sim_df,
        x_value="minutes_post_simulation",
        y_value="undelivered_basal_insulin",
        marker_color= "#f1c232",
        name="Undelivered Basal Insulin",
    ), row=4, col=1)


    # This is only updating for one of the traces
    fig.update_layout(showlegend=True,
                      paper_bgcolor='rgb(243, 243, 243)',
                      plot_bgcolor='rgb(243, 243, 243)'
                      )

    # Update xaxis properties
    for i in range(1, 5):
        fig.update_xaxes(x_axis_properties(sim_df, show_title_axis_marks=True), row=i, col=1)
        fig.update_yaxes(y_axis_properties(y_axis_labels[i - 1]), row=i, col=1)
    # fig.update_xaxes(x_axis_properties(sim_df, show_title_axis_marks=True), row=4, col=1)

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12)

    save_view_fig(fig=fig
                  , image_type=image_type
                  , figure_name=figure_name
                  , analysis_name=analysis_name
                  , view_fig=view_fig
                  , save_fig=save_fig
                  , save_fig_path= save_fig_path
                  , width=width
                  , height=height)

    return

create_simulation_figure(sim_df=simulation_example_df,
                         image_type="png",
                         figure_name="example-simulation-example-visualization",
                         analysis_name="insulin-pulses",
                         view_fig=True,
                         save_fig=True,
                         save_fig_path=os.path.join("..", "..", "reports", "figures", "insulin-pulses-risk-assessment"),
                         width=600,
                         height=900)
