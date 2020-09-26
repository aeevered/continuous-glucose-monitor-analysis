# %% REQUIRED LIBRARIES
import os
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio
import datetime as dt
import itertools
from save_view_fig import save_view_fig
import tarfile
import json
from scipy import stats
import tidepool_data_science_metrics as metrics

## Functions from data-science-metrics (ultimately want to just pull in that repo)
def approximate_steady_state_iob_from_sbr(
    scheduled_basal_rate: np.float64,
) -> np.float64:
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
    iob_array: "np.ndarray[np.float64]",
    scheduled_basal_rate: np.float64,
    round_to_n_digits: int = 3,
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
    hours_with_less_50percent_sbr_iob = (
        np.sum(indices_with_less_50percent_sbr_iob) * 5 / 60
    )

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
        warnings.warn(
            "Some values in the passed in array had glucose values less than 38."
        )

    if (bg_array > 402).any():
        warnings.warn(
            "Some values in the passed in array had glucose values greater than 402."
        )

    if (bg_array < 1).any():
        warnings.warn(
            "Some values in the passed in array had glucose values less than 1."
        )
        # raise Exception("Some values in the passed in array had glucose values less than 1.")

    if (bg_array > 1000).any():
        warnings.warn(
            "Some values in the passed in array had glucose values less than 1."
        )
        # raise Exception("Some values in the passed in array had glucose values greater than 1000.")


########### New Code ##################


#Todo: update these functions
def add_error_fields(df):
    # default icgm and ysi ranges [40, 400] and [0, 900]
    sensor_bg_range = (40, 400)
    bg_range = (0, 900)
    sensor_min, sensor_max = sensor_bg_range
    bg_min, bg_max = bg_range

    # calculate the icgm error (difference and percentage)
    sensor_bg_values = df["bg_sensor"].values
    bg_values = df["bg"].values
    icgm_error = sensor_bg_values - bg_values

    df["icgmError"] = icgm_error
    abs_difference_error = np.abs(icgm_error)
    df["absError"] = abs_difference_error
    df["absRelDiff"] = 100 * abs_difference_error / bg_values

    df["withinMeasRange"] = (
            (sensor_bg_values >= sensor_min) & (sensor_bg_values < sensor_max)
    )

    return df

def calc_mbe(df):

    # default icgm and ysi ranges [40, 400] and [0, 900]

    df = add_error_fields(df)
    return np.mean(df.loc[df["withinMeasRange"], "icgmError"])

def calc_mard(df):
    ''' Mean Absolute Relative Deviation (MARD)
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5375072/
    '''

    df = add_error_fields(df)

    abs_relative_difference_in_measurement_range = (
        df.loc[df["withinMeasRange"], "absRelDiff"]
    )
    return np.mean(abs_relative_difference_in_measurement_range)

def get_data(filename, simulation_df, patient_characteristics_df, sensor_characteristics_df):
    virtual_patient_num = filename.split("/")[-1].split(".")[0].replace("vp","")
    age = patient_characteristics_df["age"].iloc[0]
    ylw = patient_characteristics_df["ylw"].iloc[0]
    cir = simulation_df["cir"].iloc[0]
    isf = simulation_df["isf"].iloc[0]
    sbr = simulation_df["sbr"].iloc[0]

    #Todo: fill in rest of these
    bias_factor = sensor_characteristics_df["bias_norm_factor"].iloc[0]
    bias_drift_oscillations = sensor_characteristics_df["bias_drift_oscillations"].iloc[0]
    bias_drift_range_start = sensor_characteristics_df["bias_drift_range_start"].iloc[0]
    bias_drift_range_end = sensor_characteristics_df["bias_drift_range_end"].iloc[0]
    noise_coefficient = sensor_characteristics_df["noise_coefficient"].iloc[0]
    mard = calc_mard(simulation_df)
    mbe = calc_mbe(simulation_df)

    bg_test_condition = filename.split(".")[1].replace("bg", "")
    analysis_type = filename.split(".")[3]
    LBGI = metrics.glucose.blood_glucose_risk_index(bg_array=simulation_df["bg"])[0]
    LBGI_RS = metrics.glucose.lbgi_risk_score(LBGI)
    DKAI = metrics.insulin.dka_index(simulation_df["iob"], simulation_df["sbr"].iloc[0])
    DKAI_RS = metrics.insulin.dka_risk_score(DKAI)
    HBGI = metrics.glucose.blood_glucose_risk_index(bg_array=simulation_df["bg"])[1]
    return [virtual_patient_num, age, ylw, cir, isf, sbr, bias_factor, bias_drift_oscillations, bias_drift_range_start,
            bias_drift_range_end, noise_coefficient, mard, mbe, bg_test_condition, analysis_type, LBGI, LBGI_RS, DKAI, DKAI_RS, HBGI]


# %% Visualization Functions
# %% FUNCTIONS
# TODO: us mypy and specify the types

utc_string = dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%m-%S")
# TODO: automatically grab the code version to add to the figures generated
code_version = "v0-1-0"

# adding in some generic methods for tables based on bins
def bin_data(bin_breakpoints):
    # the bin_breakpoints are the points that are greater than or equal to
    return pd.IntervalIndex.from_breaks(bin_breakpoints, closed="left")


def get_metadata_tables(demographic_df):
    # %% prepare demographic data for tables

    virtual_patient_group = demographic_df.groupby("virtual_patient_num")
    demographic_reduced_df = virtual_patient_group[
        ["age", "ylw", "CIR", "ISF", "SBR"]
    ].median()

    # get replace age and years living with (ylw) < 0 with np.nan
    demographic_reduced_df[demographic_reduced_df < 0] = np.nan


    # %% Age Breakdown Table
    # TODO: this can be generalized for any time we want to get counts by bins
    age_bin_breakpoints = np.array([0, 7, 14, 25, 50, 100])
    age_bins = bin_data(age_bin_breakpoints)

    # make an age table
    age_table = pd.DataFrame(index=age_bins.astype("str"))
    age_table.index.name = "Age (years old)"

    # cut the data by bin
    demographic_reduced_df["age_bin"] = pd.cut(demographic_reduced_df["age"], age_bins)
    age_table["Count"] = demographic_reduced_df.groupby("age_bin")["age"].count().values

    # add in missing data
    age_table.loc["Missing", "Count"] = demographic_reduced_df["age"].isnull().sum()

    # make sure that counts add up correctly
    # TODO: make a test that checks that the total subjects equal the total counts in the table
    total_virtual_subjects_from_table = age_table["Count"].sum()
    assert total_virtual_subjects_from_table == len(demographic_reduced_df)

    # add total to end of table
    age_table.loc["Total", "Count"] = total_virtual_subjects_from_table

    age_table.reset_index(inplace=True)
    make_table(
        age_table,
        table_name="age-table",
        analysis_name="icgm-sensitivity-analysis",
        cell_height=[30],
        cell_width=[150],
        image_type="png",
        view_fig=True,
        save_fig=True,
    )

    # %% Years Living With (YLW) Breakdown Table
    ylw_bin_breakpoints = np.array([0, 1, 5, 100])
    ylw_bins = bin_data(ylw_bin_breakpoints)

    # make an ylw table
    ylw_table = pd.DataFrame(index=ylw_bins.astype("str"))
    ylw_table.index.name = "T1D Duration (years)"

    # cut the data by bin
    demographic_reduced_df["ylw_bin"] = pd.cut(demographic_reduced_df["ylw"], ylw_bins)
    ylw_table["Count"] = demographic_reduced_df.groupby("ylw_bin")["ylw"].count().values

    # add in missing data
    ylw_table.loc["Missing", "Count"] = demographic_reduced_df["ylw"].isnull().sum()

    # make sure that counts add up correctly
    # TODO: make a test that checks that the total subjects equal the total counts in the table
    total_virtual_subjects_from_table = ylw_table["Count"].sum()
    assert total_virtual_subjects_from_table == len(demographic_reduced_df)

    # add total to end of table
    ylw_table.loc["Total", "Count"] = total_virtual_subjects_from_table

    ylw_table.reset_index(inplace=True)
    make_table(
        ylw_table,
        table_name="ylw-table",
        analysis_name="icgm-sensitivity-analysis",
        cell_height=[30],
        cell_width=[200, 150],
        image_type="png",
        view_fig=True,
        save_fig=True,
    )
    # %% Carb to Insulin Ratio Table
    cir_bin_breakpoints = np.array(
        [
            demographic_reduced_df["CIR"].min(),
            5,
            10,
            15,
            20,
            25,
            demographic_reduced_df["CIR"].max() + 1,
        ]
    ).astype(int)
    cir_bins = bin_data(cir_bin_breakpoints)

    # make an cir table
    cir_table = pd.DataFrame(index=cir_bins.astype("str"))
    cir_table.index.name = "Carb-to-Insulin-Ratio"

    # cut the data by bin
    demographic_reduced_df["cir_bin"] = np.nan
    demographic_reduced_df["cir_bin"] = pd.cut(demographic_reduced_df["CIR"], cir_bins)
    cir_table["Count"] = demographic_reduced_df.groupby("cir_bin")["CIR"].count().values

    # add in missing data
    cir_table.loc["Missing", "Count"] = demographic_reduced_df["CIR"].isnull().sum()

    # make sure that counts add up correctly
    # TODO: make a test that checks that the total subjects equal the total counts in the table
    total_virtual_subjects_from_table = cir_table["Count"].sum()
    assert total_virtual_subjects_from_table == len(demographic_reduced_df)

    # add total to end of table
    cir_table.loc["Total", "Count"] = total_virtual_subjects_from_table

    cir_table.reset_index(inplace=True)
    make_table(
        cir_table,
        table_name="cir-table",
        analysis_name="icgm-sensitivity-analysis",
        cell_height=[30],
        cell_width=[200, 150],
        image_type="png",
        view_fig=True,
        save_fig=True,
    )

    # %% ISF Table
    isf_bin_breakpoints = np.array(
        [
            np.min([demographic_reduced_df["ISF"].min(), 5]),
            10,
            25,
            50,
            75,
            100,
            200,
            np.max([400, demographic_reduced_df["ISF"].max() + 1]),
        ]
    ).astype(int)
    isf_bins = bin_data(isf_bin_breakpoints)

    # make an isf table
    isf_table = pd.DataFrame(index=isf_bins.astype("str"))
    isf_table.index.name = "Insulin Sensitivity Factor"

    # cut the data by bin
    demographic_reduced_df["isf_bin"] = np.nan
    demographic_reduced_df["isf_bin"] = pd.cut(demographic_reduced_df["ISF"], isf_bins)
    isf_table["Count"] = demographic_reduced_df.groupby("isf_bin")["ISF"].count().values

    # add in missing data
    isf_table.loc["Missing", "Count"] = demographic_reduced_df["ISF"].isnull().sum()

    # make sure that counts add up correctly
    # TODO: make a test that checks that the total subjects equal the total counts in the table
    total_virtual_subjects_from_table = isf_table["Count"].sum()
    assert total_virtual_subjects_from_table == len(demographic_reduced_df)

    # add total to end of table
    isf_table.loc["Total", "Count"] = total_virtual_subjects_from_table

    isf_table.reset_index(inplace=True)
    make_table(
        isf_table,
        table_name="isf-table",
        analysis_name="icgm-sensitivity-analysis",
        cell_height=[30],
        cell_width=[250, 150],
        image_type="png",
        view_fig=True,
        save_fig=True,
    )

    # %% Basal Rate (BR) Table
    br_bin_breakpoints = np.append(
        np.arange(0, 1.5, 0.25),
        np.arange(1.5, demographic_reduced_df["SBR"].max() + 0.5, 0.5),
    )
    br_bins = bin_data(br_bin_breakpoints)

    # make an br table
    br_table = pd.DataFrame(index=br_bins.astype("str"))
    br_table.index.name = "Basal Rate"

    # cut the data by bin
    demographic_reduced_df["br_bin"] = np.nan
    demographic_reduced_df["br_bin"] = pd.cut(demographic_reduced_df["SBR"], br_bins)
    br_table["Count"] = demographic_reduced_df.groupby("br_bin")["SBR"].count().values

    # add in missing data
    br_table.loc["Missing", "Count"] = demographic_reduced_df["SBR"].isnull().sum()

    # make sure that counts add up correctly
    # TODO: make a test that checks that the total subjects equal the total counts in the table
    total_virtual_subjects_from_table = br_table["Count"].sum()
    assert total_virtual_subjects_from_table == len(demographic_reduced_df)

    # add total to end of table
    br_table.loc["Total", "Count"] = total_virtual_subjects_from_table

    br_table.reset_index(inplace=True)
    make_table(
        br_table,
        table_name="br-table",
        analysis_name="icgm-sensitivity-analysis",
        cell_height=[30],
        cell_width=[200, 150],
        image_type="png",
        view_fig=True,
        save_fig=True,
    )


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
            font_size=13,
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

    #print(table_height, table_width)

    save_view_fig(
        fig,
        image_type=image_type,
        figure_name=table_name,
        analysis_name=analysis_name,
        view_fig=view_fig,
        save_fig=save_fig,
        save_fig_path=save_fig_path,
        width=table_width,
        height=table_height
    )

    '''
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
    '''
    file_name = "{}-{}_{}_{}".format(
        analysis_name, table_name, utc_string, code_version
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
            y=table_df[metric].apply(lambda x: x + 1),
            points=False,
            color_discrete_sequence=px.colors.qualitative.T10,
            notched=notched_boxplot,
            log_y=True,
        )

    # Otherwise show separate boxplot for each breakdown category.
    else:
        table_df = table_df.sort_values([level_of_analysis])

        summary_fig = px.box(
            y=table_df[metric].apply(lambda x: x + 1),
            points=False,
            color=table_df[level_of_analysis + "_label"],
            color_discrete_sequence=px.colors.qualitative.T10,
            # can also explicitly define the sequence: ["red", "green", "blue"],
            notched=notched_boxplot,
            facet_col=table_df[level_of_analysis + "_label"],
            boxmode="overlay",
            log_y=True,
        )

    # TODO: adjust axes back to deal with adding +1 to all y values

    summary_fig.update_layout(
        title="Distribution of "
        + metric
        + " By "
        + level_of_analysis_dict[level_of_analysis],
        showlegend=True,
        # xaxis=dict(title=level_of_analysis_dict[level_of_analysis]),
        yaxis=dict(title=metric),
        plot_bgcolor="#D3D3D3",
        legend_title=level_of_analysis_dict[level_of_analysis],
    )

    summary_fig.update_yaxes(
        type=y_scale_type,
        # range=[0, ],
        tickvals=[1, 2, 3, 6, 11, 26, 51, 101, 251, 501],
        ticktext=["0", "1", "2", "5", "10", "25", "50", "100", "250", "500"],
    )

    summary_fig.update_traces(marker=dict(size=2, opacity=0.3))

    summary_fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[1].replace(" Analysis", ""))
    )

    save_view_fig(
        summary_fig,
        image_type,
        figure_name,
        analysis_name,
        view_fig,
        save_fig,
        save_fig_path,
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

        df = table_df[[metric, metric + " String"]]
        grouped_df = (
            df.groupby([metric, metric + " String"])
            .size()
            .reset_index(name="count")
            .sort_values(by=metric, ascending=True)
        )
        grouped_df["percentage"] = (
            grouped_df["count"] / grouped_df["count"].sum()
        ).apply(lambda x: "{:.1%}".format(x))

        # For adding in rows that don't exist
        metric_values = [0, 1, 2, 3, 4]
        for metric_value in metric_values:
            if not ((grouped_df[metric] == metric_value)).any():
                data = [[metric_value, score_dict[metric_value], 0.001, ""]]
                df2 = pd.DataFrame(
                    data, columns=[metric, metric + " String", "count", "percentage"]
                )
                grouped_df = pd.concat([grouped_df, df2], axis=0, ignore_index=True)

        grouped_df = grouped_df.sort_values(by=[metric], ascending=True)

        summary_fig = px.scatter(
            x=[1] * len(grouped_df[metric]),
            y=grouped_df[metric],
            size=grouped_df["count"],
            color=grouped_df[metric + " String"],
            # text=grouped_df["percentage"],
            color_discrete_map=color_dict,
            size_max=25,
        )

        for index, row in grouped_df.iterrows():
            if row["count"] >= 1:
                summary_fig.add_annotation(
                    x=1,
                    y=row[metric]
                    + 0.15
                    + float(row["percentage"].replace("%", "")) * 0.0015,
                    text=row["percentage"],
                    font=dict(size=12),
                    showarrow=False,
                )

        layout = go.Layout(
            showlegend=True,
            title="Distribution of "
            + metric
            + " Across "
            + level_of_analysis_dict[level_of_analysis],
            yaxis=dict(title=metric, tickvals=[0, 1, 2, 3, 4], range=[-0.25, 4.4]),
            xaxis=dict(
                title="", tickvals=[0, 1, 2], range=[0, 2], showticklabels=False
            ),
            plot_bgcolor="#D3D3D3",
            legend_title="Tidepool " + metric + "<br>",
            legend={"traceorder": "reversed"},
        )

    else:

        df = table_df[
            [
                level_of_analysis,
                level_of_analysis + "_label",
                metric,
                metric + " String",
            ]
        ]
        grouped_df = (
            df.groupby(
                [
                    level_of_analysis,
                    level_of_analysis + "_label",
                    metric,
                    metric + " String",
                ]
            )
            .size()
            .reset_index(name="count")
            .sort_values(
                by=[level_of_analysis, level_of_analysis + "_label", metric],
                ascending=True,
            )
        )

        sum_df = grouped_df.groupby(analysis_level)["count"].transform("sum")
        grouped_df["percentage"] = (
            grouped_df["count"].div(sum_df).apply(lambda x: "{:.1%}".format(x))
        )
        grouped_df["percentage"] = grouped_df["percentage"].apply(
            lambda x: x[: len(x) - 3] + "%" if x[len(x) - 3 :] == ".0%" else x
        )

        # For adding in rows that don't exist

        metric_values, analysis_levels, analysis_labels = (
            [0, 1, 2, 3, 4],
            grouped_df[level_of_analysis].unique(),
            grouped_df[level_of_analysis + "_label"].unique(),
        )

        for metric_value, level in itertools.product(metric_values, analysis_levels):
            if not (
                (grouped_df[metric] == metric_value)
                & (grouped_df[analysis_level] == level)
            ).any():
                data = [[level, metric_value, score_dict[metric_value], 0.001, ""]]
                df2 = pd.DataFrame(
                    data,
                    columns=[
                        level_of_analysis,
                        metric,
                        metric + " String",
                        "count",
                        "percentage",
                    ],
                )
                df2[level_of_analysis + "_label"] = df2[level_of_analysis].replace(
                    analysis_type_labels
                )
                grouped_df = pd.concat([grouped_df, df2], axis=0, ignore_index=True)

        grouped_df = grouped_df.sort_values(
            by=[level_of_analysis, level_of_analysis + "_label", metric], ascending=True
        )

        summary_fig = px.scatter(
            x=grouped_df[level_of_analysis + "_label"],
            y=grouped_df[metric],
            # text=grouped_df["percentage"],
            size=grouped_df["count"],
            color=grouped_df[metric + " String"],
            color_discrete_map=color_dict,
            # color=grouped_df["count"],
            # colorscale="RdYlGn",
            size_max=25,
        )

        if analysis_level == "bg_test_condition":
            annotation_font_size = 9
            height_parameter = 0.1
        else:
            annotation_font_size = 12
            height_parameter = 0.15

        for index, row in grouped_df.iterrows():
            if row["count"] >= 1:
                summary_fig.add_annotation(
                    x=row[level_of_analysis + "_label"],
                    y=row[metric]
                    + height_parameter
                    + float(row["percentage"].replace("%", "")) * 0.0015,
                    text=row["percentage"],
                    font=dict(size=annotation_font_size),
                    showarrow=False,
                )

        if level_of_analysis == "analysis_type":
            tickangle = 45
        else:
            tickangle = 0

        layout = go.Layout(
            showlegend=True,
            title="Distribution of "
            + metric
            + " Across "
            + level_of_analysis_dict[level_of_analysis],
            yaxis=dict(title=metric, tickvals=[0, 1, 2, 3, 4], range=[-0.25, 4.4]),
            xaxis=dict(
                title=level_of_analysis_dict[level_of_analysis],
                type="category",
                tickangle=tickangle,
            ),
            plot_bgcolor="#D3D3D3",
            legend_title="Tidepool " + metric + "<br>",
            legend={"traceorder": "reversed"},
        )

    summary_fig.update_layout(layout)

    save_view_fig(
        summary_fig,
        image_type,
        figure_name,
        analysis_name,
        view_fig,
        save_fig,
        save_fig_path,
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
            # log_x=True,
            color_discrete_sequence=px.colors.qualitative.T10,
        )

        layout = go.Layout(
            showlegend=True,
            title="Distribution of "
            + metric
            + " By "
            + level_of_analysis_dict[level_of_analysis],
            plot_bgcolor="#D3D3D3",
            xaxis=dict(title=metric),
            legend_title=level_of_analysis_dict[level_of_analysis],
        )

    else:

        df = table_df[[level_of_analysis, metric]]
        grouped_df = (
            df.groupby([level_of_analysis, metric]).size().reset_index(name="count")
        )

        if level_of_analysis == "analysis_type":
            summary_fig = px.histogram(
                x=grouped_df[metric],
                # log_x=True,
                facet_row=grouped_df[level_of_analysis],
                nbins=500,
                color_discrete_sequence=px.colors.qualitative.T10,
                color=grouped_df[level_of_analysis],
            )
        else:
            summary_fig = px.histogram(
                x=grouped_df[metric],
                # log_x=True,
                facet_col=grouped_df[level_of_analysis],
                facet_col_wrap=3,
                nbins=500,
                color_discrete_sequence=px.colors.qualitative.T10,
                color=grouped_df[level_of_analysis],
            )

        layout = go.Layout(
            showlegend=True,
            title="Distribution of "
            + metric
            + " Across "
            + level_of_analysis_dict[level_of_analysis],
            plot_bgcolor="#D3D3D3",
            # xaxis=dict(title=metric),
            legend_title=level_of_analysis_dict[level_of_analysis],
        )

    summary_fig.update_layout(layout)

    summary_fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[1].replace(" Analysis", ""))
    )

    save_view_fig(
        summary_fig,
        image_type,
        figure_name,
        analysis_name,
        view_fig,
        save_fig,
        save_fig_path,
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
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
):

    if level_of_analysis == "all":
        df = table_df[[metric]]
        distribution_df = df[metric].describe().to_frame().transpose()
        distribution_df.insert(0, "", ["All Analyses Combined"], True)
    else:
        df = table_df[[level_of_analysis, metric]]
        distribution_df = (
            df.groupby(level_of_analysis)[[metric]].describe().reset_index()
        )
        distribution_df.columns = distribution_df.columns.droplevel(0)

    if level_of_analysis == "bg_test_condition":
        distribution_df.iloc[:, 0] = distribution_df.iloc[:, 0].apply(
            lambda x: "BG Test Condition {}".format(x)
        )
        # ret = distribution_df.index.map(lambda x: "BG Test Condition {}".format(x))
        # distribution_df.insert(1, "", ret, True)
        # distribution_df.columns = distribution_df.columns.droplevel(0)

    distribution_df = distribution_df.round(2)

    distribution_df = distribution_df.rename(
        columns={
            "mean": "Mean",
            "50%": "Median",
            "std": "Standard Deviation",
            "min": "Minimum",
            "max": "Maximum",
            "count": "Number of Simulations",
        }
    )

    distribution_df = distribution_df.replace(
        "correction_bolus", "Correction Bolus Analyses"
    )
    distribution_df = distribution_df.replace("meal_bolus", "Meal Bolus Analyses")
    distribution_df = distribution_df.replace("temp_basal_only", "Temp Basal Analyses")

    make_table(
        distribution_df,
        image_type=image_type,
        table_name=table_name,
        analysis_name=analysis_name,
        cell_height=[30],
        cell_width=[240, 130, 100, 100, 100, 100, 100, 100],
        cell_header_height=[60],
        view_fig=view_fig,
        save_fig=save_fig,
        save_csv=save_csv,
        save_fig_path=save_fig_path,
    )
    return


# %% Summary Table
def prepare_results_for_summary_table(results_df):

    # %% first remove any/all iCGM sensor batches that did not meet iCGM special controls

    # summary_df_reduced = results_df[results_df["ICGM_PASS%"] == 100]
    summary_df_reduced = results_df.copy()

    # first do all analyses
    all_analyses_summary_df = get_summary_stats(
        summary_df_reduced, "All Analyses Combined"
    )

    # break up by analysis type
    # rename the analysis types
    summary_df_reduced.replace({"temp_basal_only": "Temp Basal Analysis"}, inplace=True)
    summary_df_reduced.replace(
        {"correction_bolus": "Correction Bolus Analysis"}, inplace=True
    )
    summary_df_reduced.replace({"meal_bolus": "Meal Bolus Analysis"}, inplace=True)

    for analysis_type in summary_df_reduced["analysis_type"].unique():
        temp_df = summary_df_reduced[
            summary_df_reduced["analysis_type"] == analysis_type
        ]
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

    return all_analyses_summary_df


def get_summary_stats(df, level_of_analysis_name):

    # Commented out risk score columsn pending whether want to show
    # median values for the categorical risk score measures

    # create a summary table
    # NOTE: there is a known bug with plotly tables https://github.com/plotly/plotly.js/issues/3251
    outcome_table_cols = [
        "Median LBGI<br>" "     (IQR)",  # adding in spacing because of bug
        # "Median LBGI Risk Score<br>"
        # "             (IQR)",  # adding in spacing because of bug
        "Median DKAI<br>" "     (IQR)",  # adding in spacing because of bug
        # "Median DKAI Risk Score<br>"
        # "             (IQR)",  # adding in spacing because of bug
    ]
    outcome_names = [
        "LBGI",
        "DKAI",
    ]  # ["LBGI", "LBGI Risk Score", "DKAI", "DKAI Risk Score"]
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


def make_frequency_table(
    results_df,
    image_type="png",
    table_name="<number-or-name>-frequency-table",
    analysis_name="analysis-<name>",
    cell_header_height=[60],
    cell_height=[30],
    cell_width=[200, 100, 150, 150],
    metric="LBGI",
    level_of_analysis="analysis_type",
    view_fig=True,
    save_fig=True,
    save_csv=True,
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
):
    level_of_analysis_dict = {
        "all": "All Analyses Combined",
        "analysis_type": "Analysis Type",
        "bg_test_condition": "BG Test Condition",
    }


    if level_of_analysis == "all":
        results_df_reduced = results_df[[metric + " String"]]
        frequency_df = results_df_reduced[metric + " String"].value_counts().to_frame()
        frequency_df = frequency_df.T

        #TODO: update this; not pythonic way to do
        percentage_df = frequency_df.apply(lambda x: x / x.sum(), axis=1)
        for row in range(len(frequency_df)):
            for col in range(len(frequency_df.columns)):
                frequency_df.iloc[row, col] = str(frequency_df.iloc[row, col]) + " (" + str('{:.1%}'.format(percentage_df.iloc[row, col])) + ")"
        column_names = [""]+list(color_dict.keys())

        frequency_df.insert(0, "", ["All Analyses Combined"], True)
    else:
        frequency_df = pd.crosstab(
            results_df[level_of_analysis], results_df[metric + " String"]
        ).reset_index()

        #TODO: update this; not pythonic way to do
        percentage_df = frequency_df.loc[:, frequency_df.columns != level_of_analysis].apply(lambda x: x / x.sum(), axis=1)
        for row in range(len(frequency_df)):
            for col in range(len(frequency_df.columns)-1):
                frequency_df.iloc[row, col+1] = str(frequency_df.iloc[row, col+1]) + " (" + str('{:.1%}'.format(percentage_df.iloc[row, col])) + ")"


        frequency_df = frequency_df.rename(
            columns={level_of_analysis: level_of_analysis_dict[level_of_analysis]}
        )
        column_names = [level_of_analysis_dict[level_of_analysis]] + list(
            color_dict.keys()
        )


    if level_of_analysis == "bg_test_condition":
        frequency_df.iloc[:, 0] = frequency_df.iloc[:, 0].apply(
            lambda x: "BG Test Condition {}".format(x)
        )

    # frequency_df = frequency_df.round(2)

    frequency_df = frequency_df.replace("correction_bolus", "Correction Bolus Analyses")
    frequency_df = frequency_df.replace("meal_bolus", "Meal Bolus Analyses")
    frequency_df = frequency_df.replace("temp_basal_only", "Temp Basal Analyses")

    for metric_value in score_dict.keys():
        if score_dict[metric_value] not in frequency_df.columns:
            frequency_df[score_dict[metric_value]] = "0 (0.0%)"

    frequency_df = frequency_df.reindex(columns=column_names)

    frequency_df = frequency_df.rename(
        columns={
            "Analysis Type": "",
            "BG Test Condition": ""
        }
    )

    make_table(
        frequency_df,
        image_type=image_type,
        table_name=table_name,
        analysis_name=analysis_name,
        cell_height=cell_height,
        cell_width=cell_width,
        cell_header_height=cell_header_height,
        view_fig=view_fig,
        save_fig=save_fig,
        save_csv=save_csv,
        save_fig_path=save_fig_path,
    )
    return


# Functions of cdfs


def ecdf(x):
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side="right") / x.size

    return result


def create_cdf(
    data,
    title="CDF",
    image_type="png",
    figure_name="<number-or-name>-boxplot",
    analysis_name="analysis-<name>",
    view_fig=True,
    save_fig=True,
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
):

    fig = go.Figure()
    fig.add_scatter(x=np.unique(data), y=ecdf(data)(np.unique(data)))
    fig.update_layout(title=title)

    save_view_fig(
        fig, image_type, figure_name, analysis_name, view_fig, save_fig, save_fig_path,
    )
    return

########## Spearman Correlation Coefficient Table #################
def spearman_correlation_table(results_df,
                               image_type="png",
                               table_name="spearman-correlation-table",
                               analysis_name="icgm-sensitivity-analysis",
                               cell_header_height=[60],
                               cell_height=[30],
                               cell_width=[250, 150, 150, 150, 150],
                               view_fig=True,
                               save_fig=True,
                               save_csv=True,
                               save_fig_path=os.path.join("..", "..", "reports", "figures"),
                               ):


    rows = ["bias_factor", "bias_drift_oscillations", "bias_drift_range_start", "bias_drift_range_end", "noise_coefficient", "mard", "mbe"]
    cols = ["LBGI", "LBGI Risk Score", "DKAI", "DKAI Risk Score"]

    data = {}

    for col in cols:
        row_data = []
        for row in rows:
            rho, pval = stats.spearmanr(results_df[row], results_df[col])
            row_data.append("("+str(round(rho, 3))+", "+str(round(pval, 3))+")")
        data[col] = row_data

    spearman_correlation_df = pd.DataFrame(data)
    spearman_correlation_df.insert(0, "", ['Bias Factor', 'Bias Drift Oscillations', "Bias Drift Range Start", "Bias Drift Range End", 'Noise Coefficient', 'Mean Absolute Relative Difference', "Mean Bias Error"])

    make_table(
        spearman_correlation_df,
        image_type=image_type,
        table_name=table_name,
        analysis_name=analysis_name,
        cell_height=cell_height,
        cell_width=cell_width,
        cell_header_height=cell_header_height,
        view_fig=view_fig,
        save_fig=save_fig,
        save_csv=save_csv,
        save_fig_path=save_fig_path,
    )
    return

# Function for checking distributions
def create_scatter(
        df,
        x_value="cir",
        y_value="LBGI",
        image_type="png",
        analysis_name="icgm_sensitivity_analysis",
        view_fig=True,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
):
    title = "Distribution of " + y_value + " by " + x_value
    figure_name = "distribution_" + y_value + "_" + x_value

    fig = go.Figure()
    fig.add_scatter(x=df[x_value], y=df[y_value], mode='markers')
    fig.update_layout(title=title, xaxis_title=x_value, yaxis_title = y_value)

    save_view_fig(
        fig, image_type, figure_name, analysis_name, view_fig, save_fig, save_fig_path,
    )
    return


#### LOAD IN DATA #####

data = []

path = os.path.join("..", "..", "data", "processed", "icgm-sensitivity-analysis-results-2020-09-19-nogit")

for i, filename in enumerate(os.listdir(path)): #[0:100]):
    if (filename.endswith(".csv")):
        if int(filename.split(".")[0].replace("vp", "")) not in (14, 3, 31, 35, 97): #Filter out the virtual patients outside of clinical bounds
            print(i, filename)
            simulation_df = pd.read_csv(os.path.join(path, filename))
            filename_components = filename.split(".")

            f = open(os.path.join(path, (filename_components[0] + ".json")), "r")
            json_data = json.loads(f.read())
            patient_characteristics_df = pd.DataFrame(json_data, index=['i', ])

            f = open(os.path.join(path, (filename_components[0] + "." + filename_components[1] + "." + filename_components[2] + ".json")), "r")
            json_data = json.loads(f.read())
            sensor_characteristics_df = pd.DataFrame(json_data, index=['i', ])

            # Add in the data
            data.append(get_data(filename, simulation_df, patient_characteristics_df, sensor_characteristics_df))



columns = [
    "virtual_patient_num",
    "age",
    "ylw",
    "CIR",
    "ISF",
    "SBR",
    "bias_factor",
    "bias_drift_oscillations",
    "bias_drift_range_start",
    "bias_drift_range_end",
    "noise_coefficient",
    "mard",
    "mbe",
    "bg_test_condition",
    "analysis_type",
    "LBGI",
    "LBGI Risk Score",
    "DKAI",
    "DKAI Risk Score",
    "HBGI"
]


results_df = pd.DataFrame(data, columns=columns)
results_df[["age", "ylw"]] = results_df[["age", "ylw"]].apply(pd.to_numeric)


# rename the analysis types
results_df.replace({"tempBasal": "Temp Basal Analysis"}, inplace=True)
results_df.replace({"correctionBolus": "Correction Bolus Analysis"}, inplace=True)

print(results_df["bias_factor"].unique())

########## DEFINE DICTIONARIES ###################
# primarily for renaming
score_dict = {
    0: "0 - None",
    1: "1 - Negligible",
    2: "2 - Minor",
    3: "3 - Serious",
    4: "4 - Critical",
}
color_dict = {
    "0 - None": "#0F73C6",
    "1 - Negligible": "#06B406",
    "2 - Minor": "#D0C07F",
    "3 - Serious": "#E18325",
    "4 - Critical": "#9A3A39",
}

analysis_type_labels = {
    "correction_bolus": "Correction Bolus",
    "meal_bolus": "Meal Bolus",
    "temp_basal_only": "Temp Basal Only",
}

results_df["analysis_type_label"] = results_df["analysis_type"].replace(
    analysis_type_labels
)
results_df["bg_test_condition_label"] = results_df["bg_test_condition"].replace(
    analysis_type_labels
)
results_df["DKAI Risk Score String"] = results_df["DKAI Risk Score"].replace(score_dict)
results_df["LBGI Risk Score String"] = results_df["LBGI Risk Score"].replace(score_dict)

level_of_analysis_dict = {
    "all": "All Analyses",
    "analysis_type": "Analysis Type",
    "bg_test_condition": "BG Test Condition",
}

#### CREATE FIGURES #####

#Check distributions
for x, y in itertools.product(["CIR", "ISF", "SBR"], ["LBGI", "DKAI", "HBGI"]):
    create_scatter(df=results_df, x_value=x, y_value=y)

'''
#Create Spearman Correlation Coefficient Table
spearman_correlation_table(results_df)

# Iterate through each metric and analysis_level category shown below and create boxplot
# figure with both log scale and linear scale.
metrics = ["LBGI", "DKAI"]
analysis_levels = ["bg_test_condition", "analysis_type", "all"]
y_axis_scales = ["log"]  # , "linear"]

for analysis_level, metric, axis_scale in itertools.product(
    analysis_levels, metrics, y_axis_scales
):
    make_boxplot(
        results_df,
        figure_name="boxplot-" + analysis_level + "-" + metric,
        analysis_name="icgm-sensitivity-analysis",
        metric=metric,
        level_of_analysis=analysis_level,
        notched_boxplot=False,
        y_scale_type=axis_scale,
        image_type="png",
        view_fig=False,
        save_fig=True,  # This is not working, need to figure out why
    )

    """
    make_histogram(
        results_df,
        figure_name="histogram-" + analysis_level + "-" + metric,
        analysis_name="icgm-sensitivity-analysis",
        metric=metric,
        level_of_analysis=analysis_level,
        image_type="png",
        view_fig=False,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
    )

    """
    make_distribution_table(
        results_df,
        table_name="distribution-table-" + analysis_level + "-" + metric,
        analysis_name="icgm-sensitivity-analysis",
        metric=metric,
        level_of_analysis=analysis_level,
        image_type="png",
        view_fig=False,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
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

all_analyses_summary_df = prepare_results_for_summary_table(results_df)

# make table
make_table(
    all_analyses_summary_df.reset_index(),
    table_name="summary-risk-table",
    analysis_name="icgm-sensitivity-analysis",
    cell_header_height=[60],
    cell_height=[30],
    cell_width=[200, 150, 150, 150],
    image_type="png",
    view_fig=False,
    save_fig=True,
)


########### DEMOGRAPHICS TABLE #################

#sim_results_location = os.path.join("..", "..", "data", "processed")
#simulation_file = "risk-sim-results-2020-04-13"
#file_import_path = os.path.abspath(os.path.join(sim_results_location, simulation_file))
#demographic_datapath = os.path.join(file_import_path + "-just-demographics-nogit.csv")
#demographic_df = pd.read_csv(demographic_datapath, index_col=[0])
#get_metadata_tables(demographic_df)



get_metadata_tables(results_df)


########## CDF Plots #################


metrics = ["LBGI", "DKAI", "LBGI Risk Score", "DKAI Risk Score"]

for metric in metrics:
    create_cdf(
        data=results_df[metric],
        title="CDF for " + metric,
        image_type="png",
        figure_name="cdf-" + metric,
        analysis_name="icgm-sensitivity-analysis",
        view_fig=False,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
    )


########## Proportion/Frequency Tables #################

metrics = ["LBGI Risk Score", "DKAI Risk Score"]
analysis_levels = ["bg_test_condition", "analysis_type", "all"]

for analysis_level, metric in itertools.product(analysis_levels, metrics):
    make_frequency_table(
        results_df,
        image_type="png",
        table_name="frequency-table-" + metric + "-" + analysis_level,
        analysis_name="icgm-sensitivity-analysis",
        cell_header_height=[30],
        cell_height=[30],
        cell_width=[250, 130, 135, 120, 120, 120],
        metric=metric,
        level_of_analysis=analysis_level,
        view_fig=False,
        save_fig=True,
        save_csv=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
    )
'''





