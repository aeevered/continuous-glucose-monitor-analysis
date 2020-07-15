# %% REQUIRED LIBRARIES
import os
import pandas as pd
import numpy as np
import warnings
import operator
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio
import datetime as dt
import itertools


from save_view_fig import save_view_fig


## Functions from data-science-metrics (maybe can just pull in that repository
def approximate_steady_state_iob_from_sbr(scheduled_basal_rate: np.float64) -> np.float64:
    """
    Approximate the amount of insulin-on-board from user's scheduled basal rate (sbr). This value
    comes from running the Tidepool Simple Diabetes Metabolism Model with the user's sbr for 8 hours.
    Parameters
    ----------
    scheduled_basal_rate : float
        a single value that represents the user's insulin needs
        NOTE: this needs to be updated to account for sbr schedule
    Returns
    -------
    float:
        insulin-on-board
    """
    # TODO: need test coverage here, which can be done by calling the diabetes metabolism model
    return scheduled_basal_rate * 2.111517

def blood_glucose_risk_index(
    bg_array: "np.ndarray[np.float64]", round_to_n_digits: int = 2
):
    """
    Calculate the LBGI, HBGI and BRGI within a set of glucose values from Clarke, W., & Kovatchev, B. (2009)
    Parameters
    ----------
    bg_array : ndarray
        1D array containing data with  float or int type.
    round_to_n_digits : int, optional
        The number of digits to round the result to.
    Returns
    -------
    int
        The number LBGI results.
    int
        The number HBGI results.
    int
        The number BRGI results.
    """
    _validate_bg(bg_array)
    bg_array[bg_array < 1] = 1  # this is added to take care of edge case BG <= 0
    transformed_bg = 1.509 * ((np.log(bg_array) ** 1.084) - 5.381)
    risk_power = 10 * (transformed_bg ** 2)
    low_risk_bool = transformed_bg < 0
    high_risk_bool = transformed_bg > 0
    rlBG = risk_power * low_risk_bool
    rhBG = risk_power * high_risk_bool
    lbgi = np.mean(rlBG)
    hbgi = np.mean(rhBG)
    bgri = round(lbgi + hbgi, round_to_n_digits)
    return (
        round(lbgi, round_to_n_digits),
        round(hbgi, round_to_n_digits),
        bgri,
    )

def lbgi_risk_score(lbgi: np.float64) -> int:
    """
    Calculate the Tidepool Risk Score associated with the LBGI
    https://docs.google.com/document/d/1EfIqZPsk_aF6ccm2uxO8Kv6677FIZ7SgjAAX6CmRWOM/
    Parameters
    ----------
    lbgi : float
        LBGI value calculated from BGRI
    Returns
    -------
    int
        The Tidepool LBGI Risk Score.
    """
    if lbgi > 10:
        risk_score = 4
    elif lbgi > 5:
        risk_score = 3
    elif lbgi > 2.5:
        risk_score = 2
    elif lbgi > 0:
        risk_score = 1
    else:
        risk_score = 0
    return risk_score

def dka_index(
    iob_array: "np.ndarray[np.float64]", scheduled_basal_rate: np.float64, round_to_n_digits: int = 3
):
    """
    Calculate the Tidepool DKA Index, which is the number of hours with less than 50% of the
    user's normal insulin needs, assuming that their scheduled basal rate can be used as a proxy
    for their insulin needs.
    https://docs.google.com/document/d/1zrQK7tQ3OJzjOXbwDgmQEeCdcig49F2TpJzNk2FU52k
    Parameters
    ----------
    iob_array : ndarray
        1D array containing the insulin-on-board time series with float type.
    scheduled_basal_rate : float (U/hr)
        a single value that represents the user's insulin needs
        NOTE: this needs to be updated to account for sbr schedule
    round_to_n_digits : int, optional
        The number of digits to round the result to.
    Returns
    -------
    float
        The Tidepool DKA Index in hours.
    """
    # TODO: this funciton needs to be updated to allow for multiple scheduled basal rates, AKA schedules
    steady_state_iob = approximate_steady_state_iob_from_sbr(scheduled_basal_rate)
    fifty_percent_steady_state_iob = steady_state_iob / 2
    indices_with_less_50percent_sbr_iob = iob_array < fifty_percent_steady_state_iob
    hours_with_less_50percent_sbr_iob = np.sum(indices_with_less_50percent_sbr_iob) * 5 / 60

    return round(hours_with_less_50percent_sbr_iob, round_to_n_digits)


def dka_risk_score(hours_with_less_50percent_sbr_iob: np.float64):
    """
    Calculate the Tidepool DKA Risk Score
    https://docs.google.com/document/d/1zrQK7tQ3OJzjOXbwDgmQEeCdcig49F2TpJzNk2FU52k
    Parameters
    ----------
    hours_with_less_50percent_sbr_iob : float
        calculated from dka_index
    Returns
    -------
    int
        The Tidepool DKAI Risk Score.
    """
    if hours_with_less_50percent_sbr_iob >= 21:
        risk_score = 4
    elif hours_with_less_50percent_sbr_iob >= 14:
        risk_score = 3
    elif hours_with_less_50percent_sbr_iob >= 8:
        risk_score = 2
    elif hours_with_less_50percent_sbr_iob >= 2:
        risk_score = 1
    else:
        risk_score = 0
    return risk_score

def _validate_input(lower_threshold: int, upper_threshold: int):
    if any(num < 0 for num in [lower_threshold, upper_threshold]):
        raise Exception("lower and upper thresholds must be a non-negative number")
    if lower_threshold > upper_threshold:
        raise Exception("lower threshold is higher than the upper threshold.")
    return


def _validate_bg(bg_array: "np.ndarray[np.float64]"):
    if (bg_array < 38).any():
        warnings.warn("Some values in the passed in array had glucose values less than 38.")

    if (bg_array > 402).any():
        warnings.warn("Some values in the passed in array had glucose values greater than 402.")

    if (bg_array < 1).any():
        warnings.warn("Some values in the passed in array had glucose values less than 1.")
        #raise Exception("Some values in the passed in array had glucose values less than 1.")

    if (bg_array > 1000).any():
        warnings.warn("Some values in the passed in array had glucose values less than 1.")
        #raise Exception("Some values in the passed in array had glucose values greater than 1000.")

### New Code

def get_data(filename, simulation_df):
    bg_test_condition = filename.split(".")[1].replace("bg", "")
    analysis_type = filename.split(".")[3]
    LBGI = blood_glucose_risk_index(bg_array = simulation_df["bg"])[0]
    LBGI_RS = lbgi_risk_score(LBGI)
    DKAI = dka_index(simulation_df["iob"], simulation_df["sbr"].iloc[0])
    DKAI_RS = dka_risk_score(DKAI)
    return [bg_test_condition, analysis_type, LBGI, LBGI_RS, DKAI, DKAI_RS]

# Iterate through all of the files

simulation_file_location = os.path.join("..", "..", "data", "processed", "icgm-sensitivity-analysis-results-3vpp-sample-2020-07-12")

data = []

for filename in os.listdir(simulation_file_location):
    print(filename)
    simulation_file_path = os.path.abspath(os.path.join(simulation_file_location, filename))

    simulation_df = pd.read_csv(simulation_file_path)

    #Add in the data
    data.append(get_data(filename, simulation_df))


columns = ["bg_test_condition", "analysis_type","LBGI", "LBGI Risk Score", "DKAI",  "DKAI Risk Score"]

results_df = pd.DataFrame(data, columns=columns)

# rename the analysis types
results_df.replace({"tempBasal": "Temp Basal Analysis"}, inplace=True)
results_df.replace(
    {"correctionBolus": "Correction Bolus Analysis"}, inplace=True
)

print(results_df)


#############################
# Define color dictionary
score_dict = {
    0: "0 - None<br>",
    1: "1 - Negligible<br>",
    2: "2 - Minor<br>",
    3: "3 - Serious<br>",
    4: "4 - Critical<br>",
}
color_dict = {
    "0 - None<br>": "#0F73C6",
    "1 - Negligible<br>": "#06B406",
    "2 - Minor<br>": "#D0C07F",
    "3 - Serious<br>": "#E18325",
    "4 - Critical<br>": "#9A3A39",
}
results_df["DKAI Risk Score String"] = results_df["DKAI Risk Score"].replace(score_dict)
results_df["LBGI Risk Score String"] = results_df["LBGI Risk Score"].replace(score_dict)

#############################
level_of_analysis_dict = {
    "all":'All Analyses',
    "analysis_type": "Analysis Type",
    "bg_test_condition": "BG Test Condition"
}


#############################

# %% Visualization Functions
# %% FUNCTIONS
# TODO: us mypy and specify the types

utc_string = dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%m-%S")
# TODO: automatically grab the code version to add to the figures generated
code_version = "v0-1-0"

def make_table(
    table_df,
    image_type="png",
    table_name="table-<number-or-name>",
    analysis_name="analysis-<name>",
    cell_height=[30],
    cell_width=[150],
    cell_header_height=[30],
    view_fig=True,
    save_fig=True,
    save_csv=True,
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
):
    # TODO: reduce the number of inputs to: df, style_dict, and save_dict
    table_cols = table_df.columns
    n_rows, n_cols = table_df.shape
    _table = go.Table(
        columnwidth=cell_width,
        header=dict(
            line_color="black",
            values=list(table_cols),
            fill_color="rgb(243, 243, 243)",
            align="center",
            font_size=14,
            height=cell_header_height[0],
        ),
        cells=dict(
            line_color="black",
            values=table_df[table_cols].T,
            fill_color="white",
            align="center",
            font_size=12,
            height=cell_height[0],
        ),
    )

    if len(cell_width) > 1:
        table_width = np.sum(np.asarray(cell_width))
    else:
        table_width = n_cols * cell_width[0]
    table_height = (n_rows + 1.5) * cell_height[0] + cell_header_height[0]
    table_layout = go.Layout(
        margin=dict(l=10, r=10, t=10, b=0), width=table_width, height=table_height
    )
    fig = go.Figure(data=_table, layout=table_layout)
    if view_fig:
        plot(fig)

    file_name = "{}-{}_{}_{}".format(
        analysis_name, table_name, utc_string, code_version
    )
    if save_fig:
        pio.write_image(
            fig=fig,
            file=os.path.join(save_fig_path, file_name + ".{}".format(image_type)),
            format=image_type,
        )
    if save_csv:
        table_df.to_csv(os.path.join(save_fig_path, file_name + ".csv"))

    return

def make_boxplot(
    table_df,
    image_type="png",
    figure_name="<number-or-name>-boxplot",
    analysis_name="analysis-<name>",
    metric="LBGI",
    level_of_analysis="analysis_type",
    notched_boxplot=True,
    y_scale_type="linear",
    view_fig=True,
    save_fig=True,
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
):
    """
    Create a boxplot figure.

    :param table_df: Table name of data to visualize.
    :param image_type: Image type for saving image (eg. png, jpeg).
    :param figure_name: Name of figure (for name of file for saving figure).
    :param analysis_name: Name of analysis (for name of file for saving figure).
    :param metric: Metric from table_df to visualize on the y-axis.
    :param level_of_analysis: Level of analysis breakdown ("all", "bg_test_condition", etc.) for x-axis.
    :param notched_boxplot: True if want the boxplot to be notched boxplot style.
    :param y_scale_type: Log or linear for y axis scale.
    :param view_fig: True if want to view figure.
    :param save_fig: True if want to save figure.
    :param save_fig_path: File path for where to save figure.
    :return:
    """

    # If level_of_analysis is to show all analyses (no breakdown), show as single box.
    if level_of_analysis == "all":
        summary_fig = px.box(
            x=None,
            y=table_df[metric].apply(lambda x: x+1),
            #points="all",
            color_discrete_sequence=px.colors.qualitative.T10,
            notched=notched_boxplot,
            log_y=True
        )

    # Otherwise show separate boxplot for each breakdown category.
    else:
        summary_fig = px.box(
            y=table_df[metric].apply(lambda x: x+1),
            #points = "all",
            color=table_df[level_of_analysis],
            color_discrete_sequence=px.colors.qualitative.T10,
            # can also explicitly define the sequence: ["red", "green", "blue"],
            notched=notched_boxplot,
            facet_col=table_df[level_of_analysis],
            boxmode="overlay",
            log_y=True
        )

    #TODO: adjust axes back to deal with adding +1 to all y values

    layout = go.Layout(
        title= "Distribution of " + metric + " By " + level_of_analysis_dict[level_of_analysis],
        showlegend=True,
        yaxis=dict(
            title=metric, #, type=y_scale_type,
            #range=[np.log(min(table_df[metric])+.01), np.log(max(table_df[metric])+1)]
        ),
        xaxis=dict(title=level_of_analysis_dict[level_of_analysis]),
        plot_bgcolor="#D3D3D3",
        legend_title=level_of_analysis_dict[level_of_analysis]
    )

    summary_fig.update_traces(marker=dict(size=2, opacity=0.3))

    summary_fig.update_layout(layout)

    summary_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1].replace(" Analysis", "")))

    save_view_fig(
        summary_fig, image_type, figure_name, analysis_name, view_fig, save_fig, save_fig_path
    )

    return


def make_bubble_plot(
    table_df,
    image_type="png",
    figure_name="<number-or-name>-bubbleplot",
    analysis_name="analysis-<name>",
    metric="LBGI",
    level_of_analysis="analysis_type",
    view_fig=True,
    save_fig=True,
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
):
    if level_of_analysis == "all":

        df = table_df[[metric, metric+ " String"]]
        grouped_df = df.groupby([metric, metric+ " String"]).size().reset_index(name="count").sort_values(by=metric, ascending=True)
        grouped_df['percentage'] = (grouped_df["count"] / grouped_df["count"].sum()).apply(lambda x: "{:.1%}".format(x))

        summary_fig = px.scatter(
            y=grouped_df[metric],
            size=grouped_df["count"],
            color=grouped_df[metric + " String"],
            text=grouped_df["percentage"],
            color_discrete_map=color_dict,
            size_max=25,
        )

        summary_fig.update_traces(textposition='top center', textfont_size=8)

        layout = go.Layout(
            showlegend=True,
            title="Distribution of " + metric + " Across " + level_of_analysis_dict[level_of_analysis],
            yaxis=dict(
                title=metric, tickvals=[0, 1, 2, 3, 4], range=[-.25, 4.25]
            ),
            xaxis=dict(type='category', showticklabels=False),
            plot_bgcolor="#D3D3D3",
            legend_title="Tidepool " + metric + '<br>',
            legend={'traceorder':'normal'}
        )

    else:

        df = table_df[[level_of_analysis, metric, metric+ " String"]]
        grouped_df = df.groupby([level_of_analysis, metric, metric+ " String"]).size().reset_index(name="count").sort_values(by=[metric, level_of_analysis], ascending=True)
        grouped_df['percentage'] = (grouped_df["count"] / grouped_df["count"].sum()).apply(lambda x: "{:.1%}".format(x))

        summary_fig = px.scatter(
            x=grouped_df[level_of_analysis],
            y=grouped_df[metric],
            text=grouped_df["percentage"],
            size=grouped_df["count"],
            color=grouped_df[metric+ " String"],
            color_discrete_map=color_dict,
            #color=grouped_df["count"],
            #colorscale="RdYlGn",
            size_max=25,
        )

        summary_fig.update_traces(textposition='top center', textfont_size=8)

        if level_of_analysis=="analysis_type":
            tickangle=45
        else:
            tickangle=0

        layout = go.Layout(
            showlegend=True,
            title="Distribution of " + metric + " Across " + level_of_analysis_dict[level_of_analysis],
            yaxis=dict(
                title=metric, tickvals=[0, 1, 2, 3, 4], range=[-.25, 4.25]
            ),
            xaxis=dict(title=level_of_analysis_dict[level_of_analysis], type='category', tickangle=tickangle),
            plot_bgcolor="#D3D3D3",
            legend_title="Tidepool " + metric + '<br>',
            legend={'traceorder':'normal'}
        )

    summary_fig.update_layout(layout)



    save_view_fig(
        summary_fig, image_type, figure_name, analysis_name, view_fig, save_fig, save_fig_path
    )

    return

def make_histogram(
    table_df,
    image_type="png",
    figure_name="<number-or-name>-histogram",
    analysis_name="analysis-<name>",
    metric="LBGI",
    level_of_analysis="analysis_type",
    view_fig=True,
    save_fig=True,
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
):
    if level_of_analysis == "all":

        df = table_df[[metric]]
        grouped_df = df.groupby([metric]).size().reset_index(name="count")

        summary_fig = px.histogram(
            x=grouped_df[metric],
            nbins=500,
            #log_x=True,
            color_discrete_sequence=px.colors.qualitative.T10
        )

        layout = go.Layout(
            showlegend=True,
            title="Distribution of " + metric + " By " + level_of_analysis_dict[level_of_analysis],
            plot_bgcolor="#D3D3D3",
            xaxis=dict(
                title=metric
            ),
            legend_title=level_of_analysis_dict[level_of_analysis]
        )

    else:

        df = table_df[[level_of_analysis, metric]]
        grouped_df = (
            df.groupby([level_of_analysis, metric]).size().reset_index(name="count")
        )

        if level_of_analysis == "analysis_type":
            summary_fig = px.histogram(
                x=grouped_df[metric],
                #log_x=True,
                facet_row=grouped_df[level_of_analysis],
                nbins=500,
                color_discrete_sequence=px.colors.qualitative.T10,
                color=grouped_df[level_of_analysis]
            )
        else:
            summary_fig = px.histogram(
                x=grouped_df[metric],
                #log_x=True,
                facet_col=grouped_df[level_of_analysis],
                facet_col_wrap=3,
                nbins=500,
                color_discrete_sequence=px.colors.qualitative.T10,
                color=grouped_df[level_of_analysis]
            )

        layout = go.Layout(
            showlegend=True,
            title="Distribution of " + metric + " Across " + level_of_analysis_dict[level_of_analysis],
            plot_bgcolor="#D3D3D3",
            #xaxis=dict(title=metric),
            legend_title=level_of_analysis_dict[level_of_analysis],
        )

    summary_fig.update_layout(layout)

    summary_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1].replace(" Analysis", "")))

    save_view_fig(
        summary_fig, image_type, figure_name, analysis_name, view_fig, save_fig, save_fig_path
    )

    return


def make_distribution_table(
        table_df,
        image_type="png",
        table_name="<number-or-name>-table",
        analysis_name="analysis-<name>",
        metric="LBGI",
        level_of_analysis="analysis_type",
        view_fig=True,
        save_fig=True,
        save_csv=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures")
):

    header = [metric]+[""]*8


    if level_of_analysis == "all":
        df = table_df[[metric]]
        distribution_df = df[metric].describe().to_frame().transpose()
        print(distribution_df)
        header = [metric] + [""] * 7
    else:
        df = table_df[[level_of_analysis, metric]]
        distribution_df = df.groupby(level_of_analysis)[[metric]].describe().reset_index()
        header = [metric] + [""] * 8

    distribution_df = distribution_df.round(2)

    #distribution_df.iloc[-1] = header

    make_table(
        distribution_df,
        image_type=image_type,
        table_name=table_name,
        analysis_name=analysis_name,
        cell_height=[30],
        cell_width=[150],
        cell_header_height=[30],
        view_fig=view_fig,
        save_fig=save_fig,
        save_csv=save_csv,
        save_fig_path=save_fig_path,
    )
    return

# Iterate through each metric and analysis_level category shown below and create boxplot
# figure with both log scale and linear scale.
metrics = ["LBGI", "DKAI"]
analysis_levels = ["bg_test_condition", "analysis_type", "all"]
y_axis_scales = ["log"] #, "linear"]

for analysis_level, metric, axis_scale in itertools.product(
    analysis_levels, metrics, y_axis_scales
):
    make_boxplot(
        results_df,
        figure_name="boxplot-" + analysis_level + "-" + metric,
        analysis_name="icgm-sensitivity-analysis",
        metric=metric,
        level_of_analysis=analysis_level,
        notched_boxplot=True,
        y_scale_type=axis_scale,
        image_type="png",
        view_fig=False,
        save_fig=True  # This is not working, need to figure out why
    )

    make_histogram(
        results_df,
        figure_name="histogram-" + analysis_level + "-" + metric,
        analysis_name="icgm-sensitivity-analysis",
        metric=metric,
        level_of_analysis=analysis_level,
        image_type="png",
        view_fig=False,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures")
    )

    make_distribution_table(
        results_df,
        table_name="distribution-table-" + analysis_level + "-" + metric,
        analysis_name="icgm-sensitivity-analysis",
        metric=metric,
        level_of_analysis=analysis_level,
        image_type="png",
        view_fig=False,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures")
    )


metrics = ["LBGI Risk Score", "DKAI Risk Score"]
analysis_levels = ["bg_test_condition", "analysis_type", "all"]

for analysis_level, metric in itertools.product(analysis_levels, metrics):
    make_bubble_plot(
        results_df,
        image_type="png",
        figure_name="bubbleplot-" + analysis_level + "-" + metric,
        analysis_name="icgm-sensitivity-analysis",
        metric=metric,
        level_of_analysis=analysis_level,
        view_fig=False,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
    )


########### SUMMARY TABLE #################

# %% Summary Table
def get_summary_stats(df, level_of_analysis_name):

    # create a summary table
    # NOTE: there is a known bug with plotly tables https://github.com/plotly/plotly.js/issues/3251
    outcome_table_cols = [
        "Median LBGI<br>" "     (IQR)",  # adding in spacing because of bug
        "Median LBGI Risk Score<br>"
        "             (IQR)",  # adding in spacing because of bug
        "Median DKAI<br>" "     (IQR)",  # adding in spacing because of bug
        "Median DKAI Risk Score<br>"
        "             (IQR)",  # adding in spacing because of bug
    ]
    outcome_names = ["LBGI", "LBGI Risk Score", "DKAI", "DKAI Risk Score"]
    count_name = " Number of<br>Simulations"
    summary_table_cols = [count_name] + outcome_table_cols
    summary_table = pd.DataFrame(columns=summary_table_cols)
    summary_table.index.name = "Level of Analysis"

    for outcome, outcome_table_col in zip(outcome_names, outcome_table_cols):
        summary_stats = pd.Series(df[outcome].describe())
        summary_table.loc[level_of_analysis_name, count_name] = summary_stats["count"]
        summary_table.loc[
            level_of_analysis_name, outcome_table_col
        ] = "{} (IQR={}-{})".format(
            summary_stats["50%"].round(1),
            summary_stats["25%"].round(1),
            summary_stats["75%"].round(1),
        )
    return summary_table


# %% first remove any/all iCGM sensor batches that did not meet iCGM special controls
#summary_df_reduced = results_df[results_df["ICGM_PASS%"] == 100]
summary_df_reduced = results_df.copy()

# first do all analyses
all_analyses_summary_df = get_summary_stats(summary_df_reduced, "All Analyses Combined")

# break up by analysis type
# rename the analysis types
summary_df_reduced.replace({"tempBasal": "Temp Basal Analysis"}, inplace=True)
summary_df_reduced.replace(
    {"correctionBolus": "Correction Bolus Analysis"}, inplace=True
)
summary_df_reduced.replace({"mealBolus": "Meal Bolus Analysis"}, inplace=True)

for analysis_type in summary_df_reduced["analysis_type"].unique():
    temp_df = summary_df_reduced[summary_df_reduced["analysis_type"] == analysis_type]
    temp_summary = get_summary_stats(temp_df, analysis_type)
    all_analyses_summary_df = pd.concat([all_analyses_summary_df, temp_summary])

# break up by bg test condition
summary_df_reduced = summary_df_reduced.sort_values(by=["bg_test_condition"])
for bg_test_condition in summary_df_reduced["bg_test_condition"].unique():
    temp_df = summary_df_reduced[
        summary_df_reduced["bg_test_condition"] == bg_test_condition
    ]
    temp_summary = get_summary_stats(
        temp_df, "BG Test Condition {}".format(bg_test_condition)
    )


    all_analyses_summary_df = pd.concat([all_analyses_summary_df, temp_summary])

# make table
make_table(
    all_analyses_summary_df.reset_index(),
    table_name="summary-risk-table",
    analysis_name="icgm-sensitivity-analysis",
    cell_header_height=[60],
    cell_height=[30],
    cell_width=[200, 100, 150, 200, 150, 200],
    image_type="png",
    view_fig=True,
    save_fig=True,
)