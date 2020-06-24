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

insulin_pulse_file_location = os.path.join("..", "..", "data", "raw", "insulin-pulses-risk-assessment")
summary_metrics_file = "insulin-pulses-risk-assessment-summary-nogit.csv"
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

    for value in table_df[color_value_column].unique():
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
                size=10,
                line_width=1
            )
        ))

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=x_title,
            gridcolor='rgb(255, 255, 255)',
            gridwidth=2),
        yaxis=dict(
            title=y_title,
            gridcolor='rgb(255, 255, 255)',
            gridwidth=2,
            type='linear'),
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
        title="DKAI Risk Score by Basal Rate",
        x_title="Scheduled Basal Rate",
        y_title="DKAI Risk Score",
        image_type="png",
        figure_name="summary-metrics-dkai-riskscore-scatterplot",
        analysis_name="insulin-pulses",
        view_fig=True,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures", "insulin-pulses-risk-assessmet")
)


# Simulation Example Plot
# See here for example figures:
# https://colab.research.google.com/drive/1oVWDc734_RndivI5lAcc8dqxtwft5kMA