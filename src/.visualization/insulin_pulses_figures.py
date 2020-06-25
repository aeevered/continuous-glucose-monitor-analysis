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
#Use the 5 colors sketched out in this doc:
# https://docs.google.com/document/d/1XBUKXTzhWR49ZS63mDXlkTePFK1KDmpRZsPtz3_J06E/edit?ts=5ef2527a#
#Cameron did a heatmap, but I think we can do a plotly scattergl plot.
#BONUS: add the DKA index value to the hover state


#from make_figures_and_tables import save_view_fig

def save_view_fig(
        fig,
        image_type="png",
        figure_name="<number-or-name>",
        analysis_name="analysis-<name>",
        view_fig=True,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
):
    if view_fig:
        fig.show()

    file_name = "{}-{}_{}_{}".format(analysis_name, figure_name, utc_string, code_version)

    if save_fig:
        pio.write_image(
            fig=fig, file=os.path.join(save_fig_path, file_name + ".{}".format(image_type)), format=image_type
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
):
    traces = []

    for value in sorted(table_df[color_value_column].unique()):
        df = table_df[table_df[color_value_column]==value]

        traces.append(go.Scattergl(
            x=df[x_value],
            y=df[y_value],
            mode='markers',
            hovertext=table_df[hover_value],
            name=str(value) + " - " + score_dict[value],
            showlegend=True,
            marker=dict(
                color=[color_dict[value]]*len(df.index),
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

    save_view_fig(fig, image_type, figure_name, analysis_name, view_fig, save_fig, save_fig_path)

    return

#Create Scatterplot Figure

#Define color dictionary
color_dict = {0:"#0F73C6",
              1:"#06B406",
              2:"#D0C07F",
              3:"#E18325",
              4:"#9A3A39"}

score_dict = {0:"None",
              1:"Negative",
              2:"Minor",
              3:"Serious",
              4:"Critical"}

create_scatterplot(
        table_df=summary_metrics_df,
        x_value="sbr",
        y_value="loop_max_basal_rate",
        hover_value="dka_index",
        color_value_column="dka_risk_score",
        color_dict=color_dict,
        score_dict=score_dict,
        legend_title = "DKAI Risk Score",
        title="DKAI Risk Score by Basal Rate",
        x_title="Scheduled Basal Rate",
        y_title="Loop Max Allowable Basal Rate",
        image_type="png",
        figure_name="summary-metrics-dkai-riskscore-scatterplot",
        analysis_name="insulin-pulses",
        view_fig=True,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures", "insulin-pulses-risk-assessment")
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
        style

):
    trace=go.Scatter(
        x=x_value,
        y=y_value,
        hoverinfo="y+name+x",
        name = name,
        mode=style,
        line_shape='hv',
        marker=dict(
            size=4,
            line=dict(width=0),
            color=color,
            symbol=symbol
        )
    )

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


def create_simulation_figure(sim_df):
    sim_df['five_minute_marks'] = sim_df.index
    sim_df['minutes_post_simulation'] = sim_df['five_minute_marks'].apply(lambda x: x*5)

    fig = make_subplots(rows=5
                        , cols=1
                        , subplot_titles=("BG Over Time", "Insulin Delivery", "Delivered Basal Insulin", "Undelivered Basal Insulin")
                        , vertical_spacing=0.15)

    y_fields = ["bg", "sbr", 'temp_basal','bolus', 'delivered_basal_insulin', 'undelivered_basal_insulin']
    colors = ["#D9CEED", "#008ECC","#008ECC","#008ECC", "#008ECC","#008ECC"]
    names = ["bg", "sbr", 'temp_basal','bolus', 'delivered_basal_insulin', 'undelivered_basal_insulin']
    symbols = ['circle', 'circle', 'triangle-up-dot', 'triangle-up-dot', 'triangle-up-dot', 'triangle-up-dot']
    styles = ['markers', 'lines', 'lines', 'lines', 'lines', 'lines']
    y_axis_labels = ["BG (mg/dL)", "Insulin (U or U/hr)", "Insulin (U)", "Insulin(U)"]
    #dash = []
    #line_shape=[line_shape='hv']
    rows = [1, 2, 2, 2, 3, 4]


    for y_field, color, name, symbol, row, style in zip(y_fields, colors, names, symbols, rows, styles):
        trace = make_scatter_trace(
            x_value=sim_df["minutes_post_simulation"],
            y_value=sim_df[y_field],
            color=color,
            symbol=symbol,
            name=name,
            style=style

        )
        fig.append_trace(trace, row=row, col=1)

    #This is only updating for one of the traces
    fig.update_layout(showlegend=True,
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        height=900,
        width=600,
        hovermode="x",

    )

    # Update xaxis properties
    for i in range(1, 5):
        fig.update_xaxes(x_axis_properties(sim_df, show_title_axis_marks=True), row=i, col=1)
        fig.update_yaxes(y_axis_properties(y_axis_labels[i-1]), row=i, col=1)
    #fig.update_xaxes(x_axis_properties(sim_df, show_title_axis_marks=True), row=4, col=1)

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12)

    fig.show()

create_simulation_figure(simulation_example_df)

'''
def make_sim_figure(
        sim_df,
):
    traces = []
    traces.append(make_bg_trace(sim_df = sim_df,
                                , x_value = "time"
                                , y_value = "bg_sensor"
                                , name = "Sensor BG"
                                , color = ""
                                , symbol =

def make_bg_trace(sim_df,
                  x_value,
                  y_value,
                  name,
                  color,
                  symbol):
    bg_trace = go.Scattergl(
        name=name,
        x=sim_df[x_value],
        y=sim_df[y_value],
        hoverinfo="y+name+x",
        mode='markers',
        marker=dict(
            size=4,
            line=dict(width=0),
            color=color,
            symbol=symbol
        )
    )

    return bg_trace


def prepare_bg(sim_df):
    df_axis = dict(
        domain=[0.4, 1],
        # range=[1.6, 2.8], # log range
        range=[0, 400],  # non-log range
        # type = "log",
        # tickvals=[-100, 54, 70, 140, 180, 250, 400],
        fixedrange=True,
        dtick=50,
        hoverformat=".0f",
        zeroline=False,
        showgrid=True,
        gridcolor="#c0c0c0",
        title=dict(
            text="rTBG<br>(mg/dL)",
            font=dict(
                size=12
            )
        )
    )

    df_annotations = go.layout.Annotation(
        x=0,
        y=250,  # 250 linear range
        xref="x",
        yref="y2",
        text="simulation start (t=0)",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ayref="y2",
        ay=320,  # linear range
        arrowcolor="black"
    )

    bg_fields = ["seed_trueBG", "seed_iCGM", 'bg_actual', 'bg_loop']
    bg_colors = ["#D9CEED", "#6aa84f", "#D9CEED", "#6aa84f"]
    bg_names = ["scenario input rTBG Trace", "scenario input iCGM Trace", "simulated output rTBG Trace",
                "simulated output iCGM Trace"]
    bg_symbols = ['circle', 'circle', 'triangle-up-dot', 'triangle-up-dot']
    bg_traces = []

    for field, name, color, symbol in zip(bg_fields, bg_names, bg_colors, bg_symbols):
        bg_trace = make_bg_trace(field, sim_df, name, color, symbol)
        bg_trace.yaxis = "y2"
        bg_traces.append(bg_trace)

    return bg_traces, df_axis, df_annotations


##For basal/insulin traces
def make_basal_trace(field, sim_df, name):
    if "delivered" in field:
        dash = "solid"
        width = 1.3
        fill = 'tozeroy'
        opacity = 0.25
        color = "#008ECC"
        line_shape = 'vh'
    elif field == 'iob':
        dash = "solid"
        width = 1.7
        fill = 'none'
        opacity = 0.75
        color = "#008ECC"
        line_shape = 'spline'
    elif field == "BR":
        dash = "dot"
        width = 1.3
        fill = 'none'
        opacity = 0.75
        color = "#008ECC"
        line_shape = 'vh'

    basal_trace = go.Scatter(
        name=name,
        showlegend=True,
        mode='lines',
        x=sim_df["hours"],
        y=sim_df[field],
        hoverinfo="none",
        line=dict(
            shape='vh',
            color=color,
            dash=dash,
            width=width,
        ),
        line_shape=line_shape,
        fill=fill,
        fillcolor='rgba(86,145,240,{})'.format(opacity)
    )

    return basal_trace


def prepare_basal(sim_df):
    max_value = max(sim_df["insulinLoop"].max() + 1,
                    sim_df["iob"].max() + 1,
                    sim_df["temp_basal"].max() + 2
                    )

    basal_axis = dict(
        domain=[0.1, 0.35],
        range=[0, max_value],
        # type = "log",
        fixedrange=True,
        hoverformat=".2f",
        showgrid=True,
        gridcolor="#c0c0c0",
        title=dict(
            text="Events<br>(U, U/hr)",
            font=dict(
                size=11
            )
        )
    )

    basal_annotation = go.layout.Annotation(
        x=0,
        y=.5,
        xref="x",
        yref="y",
        text="simulation start (t=0)",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ayref="y",
        ay=-3
    )

    basal_fields = ['iob', 'basal_delivered', "BR"]
    basal_names = ['insulin-on-board', 'basal insulin delivered', 'scheduled basal rate']
    basal_traces = []

    for field, name in zip(basal_fields, basal_names):
        basal_trace = make_basal_trace(field, sim_df, name)
        basal_trace.yaxis = "y"
        basal_traces.append(basal_trace)

    return basal_traces, basal_axis, basal_annotation


def prepare_carbs(sim_df):
    # drop rows where grams is null
    carb_df = sim_df[sim_df['carbLoop'].notna()]
    carb_df = carb_df[carb_df['carbLoop'] != 0]

    carb_df["bolus_height"] = (
            carb_df["carbLoop"] / carb_df["CIR"]
    )

    carb_trace = go.Scatter(
        name="carbs (g)",
        mode='markers + text',
        x=carb_df["hours"],
        y=carb_df["insulinLoop"] * 1.2,
        hoverinfo="name+x",
        marker=dict(
            color="#f1c232",
            size=22
        ),
        opacity=0.75,
        text=carb_df["carbLoop"],
        textposition='middle center'
    )

    carb_trace.yaxis = "y"

    return carb_trace


def prepare_bolus(sim_df):
    # drop rows where bolus is null or zero
    bolus_df = sim_df[sim_df['insulinLoop'].notna()]
    bolus_df = bolus_df[bolus_df['insulinLoop'] != 0]

    df_trace = go.Scatter(
        name="bolus",
        mode='markers + text',
        x=bolus_df["hours"],
        y=bolus_df["insulinLoop"],
        hoverinfo="name+x",
        marker=dict(
            color="#0079C0",
            size=20,
            symbol="triangle-down",
            line=dict(
                color='white',
                width=2
            )
        ),
        opacity=0.50,
        textfont=dict(
            color="black",
            size=10
        ),
        text=bolus_df["insulinLoop"],
        textposition='bottom center'
    )

    df_trace.yaxis = "y"

    return df_trace
'''
''' 
def prepare_layout(sim_df
                   , hours_pre_simulation
                   , hours_post_simulation
                   , bg_axis
                   , insulin_axis
                   , top_annotation
                   , bottom_annotation):
    layout = go.Layout(
        showlegend=True,
        plot_bgcolor="white",
        yaxis2=bg_axis,
        yaxis=insulin_axis,

        xaxis=dict(
            range=(
                -hours_pre_simulation,  # -5*hours_pre_simulation*12,
                hours_post_simulation  # 5*hours_post_simulation*12
            ),
            showgrid=True,
            gridcolor="#c0c0c0",
            hoverformat="%H:%M",
            tickmode='linear',
            dtick=1,
            title=dict(
                text="Hours (Pre and Post Simulation)",
                font=dict(
                    size=12
                )
            )
        ),
        annotations=[
            top_annotation
            # ,bottom_annotation
        ],
        dragmode="pan",
        hovermode="x"
    )

    return layout


def make_scenario_figure(sim_df, hours_pre_simulation, hours_post_simulation):
    # %% blood glucose data
    bg_traces, bg_axis, bg_annotation = (
        prepare_bg(sim_df)
    )

    # %% insulin data

    # prepare insulin axis

    # basal data
    basal_traces, insulin_axis, insulin_annotation = (
        prepare_basal(sim_df)
    )

    # carb data (cir and carb events)
    #carb_trace = prepare_carbs(sim_df)

    # bolus data
    #bolus_trace = prepare_bolus(sim_df)

    # get plot title
    title = get_plot_title(sim_df)

    # %% make figure
    fig_layout = prepare_layout(sim_df,
                                hours_pre_simulation, hours_post_simulation,
                                bg_axis, insulin_axis
                                ,bg_annotation, insulin_annotation
                                )

    traces = []
    traces.extend(bg_traces)
    traces.extend(basal_traces)
    traces.extend([carb_trace, bolus_trace])

    fig = go.Figure(data=traces, layout=fig_layout)

    fig.update_layout(
        title=title,
        font=dict(
            size=9
        )
    )

    return fig


def plot_figure(sim_data, simulation_number, hours_pre_simulation, hours_post_simulation):
    sim_df = prepare_data(sim_data, simulation_number, hours_pre_simulation, hours_post_simulation)
    plotly_fig = make_scenario_figure(sim_df, hours_pre_simulation, hours_post_simulation)
    iplot(plotly_fig)

plot_figure(sim_data, simulation_number, HOURS_PRE_SIMULATION, HOURS_POST_SIMULATION)
'''