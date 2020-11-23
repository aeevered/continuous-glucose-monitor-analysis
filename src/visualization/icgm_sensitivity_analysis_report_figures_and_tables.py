__author__ = "Anne Evered"

# %% REQUIRED LIBRARIES
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
import itertools
from src.visualization.save_view_fig import save_view_fig
import json
from scipy import stats
import tidepool_data_science_metrics as metrics
from plotly.subplots import make_subplots


# Todo: Update to use MARD and MBE functions from icgm-sensitivity-analysis or from data-science-metrics Reference:
#  https://github.com/tidepool-org/icgm-sensitivity-analysis/blob/jameno/analysis-tables/src/simulator_functions.py
# from tidepool_data_science_models.models.icgm_sensor_generator_functions import (
#     calc_mard,
#     preprocess_data,
#     calc_mbe,
#     calc_icgm_sc_table,
#     calc_icgm_special_controls_loss,
# )


def add_error_fields(df):
    """

    Parameters
    ----------
    df: dataframe
        dataframe to add error fields to (for use in MARD and MBE calculations)

    Returns
    -------
    df: dataframe
        dataframe with new error field columns

    """
    # Default iCGM and ysi ranges [40, 400] and [0, 900]
    sensor_bg_range = (40, 400)
    sensor_min, sensor_max = sensor_bg_range

    # Calculate the iCGM error (difference and percentage)
    sensor_bg_values = df["bg_sensor"].values
    bg_values = df["bg"].values
    icgm_error = sensor_bg_values - bg_values

    # Add error field to dataframe
    df["icgmError"] = icgm_error
    abs_difference_error = np.abs(icgm_error)
    df["absError"] = abs_difference_error
    df["absRelDiff"] = 100 * abs_difference_error / bg_values

    df["withinMeasRange"] = (sensor_bg_values >= sensor_min) & (
        sensor_bg_values <= sensor_max
    )

    return df


def calc_mbe(df):
    """

    Calculate mean bias

    Parameters
    ----------
    df: dataframe
        dataframe to calculate mean bias error (MBE) from

    Returns
    -------
    mean bias error calculation

    """
    df = add_error_fields(df)
    return np.mean(df.loc[df["withinMeasRange"], "icgmError"])


def calc_mard(df):
    """
    Calculate Mean Absolute Relative Deviation (MARD)
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5375072/

    Parameters
    ----------
    df: dataframe
        dataframe to calculate mean absolute relative deviation (MARD) from

    Returns
    -------
    mard calculation

    """
    df = add_error_fields(df)

    abs_relative_difference_in_measurement_range = df.loc[
        df["withinMeasRange"], "absRelDiff"
    ]

    return np.mean(abs_relative_difference_in_measurement_range)


# TODO: this may not be the best way of loading the data files in because of speed and
# and could be prone to error since need to make sure the fields returning and the column names match


def get_data(
    filename, simulation_df, simulation_characteristics_json_data, baseline=False
):
    """

    Returns a list of data

    Parameters
    ----------
    filename: str
        name of file corresponding
    simulation_df: dataframe
        dataframe of the particular simulation want to return data for
    simulation_characteristics_json_data: dataframe
        json simulation characteristics data corresponding to that simulaton
    baseline: bool
        whether this particular file is a baseline file

    Returns
    -------
    list of data items that will be a row in aggregated summary dataframe

    """
    sim_id = simulation_characteristics_json_data["sim_id"]
    virtual_patient_num = simulation_characteristics_json_data["sim_id"].split(".")[0]

    # Parse sensor_num and patient_scenario_filename from filename
    sensor_num = filename.split(".")[2]
    patient_scenario_filename = filename.split(".")[0]

    # Get simulation characteristics from json data
    age = simulation_characteristics_json_data["controller"]["config"]["age"]
    ylw = simulation_characteristics_json_data["controller"]["config"]["ylw"]
    cir = simulation_characteristics_json_data["patient"]["config"][
        "carb_ratio_schedule"
    ]["schedule"][0]["setting"].replace(" g", "")
    isf = simulation_characteristics_json_data["patient"]["config"][
        "insulin_sensitivity_schedule"
    ]["schedule"][0]["setting"].replace(" m", "")
    sbr = simulation_characteristics_json_data["patient"]["config"]["basal_schedule"][
        "schedule"
    ][0]["setting"].replace(" U", "")

    starting_bg = simulation_df["bg"].iloc[0]
    starting_bg_sensor = simulation_df["bg_sensor"].iloc[0]
    true_bolus = simulation_df["true_bolus"].iloc[1]

    # If the file is a baseline file, set all sensor characteristics to nan
    # These columns are included in case there is a future case where the baseline
    # sensors have sensor characteristics and do not just return the bg value.
    if baseline:
        initial_bias = np.nan
        bias_norm_factor = np.nan
        bias_drift_oscillations = np.nan
        bias_drift_range_start = np.nan
        bias_drift_range_end = np.nan
        noise_coefficient = np.nan
        delay = np.nan
        bias_drift_type = np.nan
        bias_type = np.nan
        noise_per_sensor = np.nan
        noise = np.nan
        bias_factor = np.nan
        phi_drift = np.nan
        drift_multiplier = np.nan
        drift_multiplier_start = np.nan
        drift_multiplier_end = np.nan
        noise_max = np.nan
        mard = np.nan
        mbe = np.nan

    # If the file is not a baseline file, get sensor characteristics from the json data
    else:
        initial_bias = simulation_characteristics_json_data["patient"]["sensor"][
            "initial_bias"
        ]
        bias_norm_factor = simulation_characteristics_json_data["patient"]["sensor"][
            "bias_norm_factor"
        ]
        bias_drift_oscillations = simulation_characteristics_json_data["patient"][
            "sensor"
        ]["bias_drift_oscillations"]
        bias_drift_range_start = simulation_characteristics_json_data["patient"][
            "sensor"
        ]["bias_drift_range_start"]
        bias_drift_range_end = simulation_characteristics_json_data["patient"][
            "sensor"
        ]["bias_drift_range_end"]
        noise_coefficient = simulation_characteristics_json_data["patient"]["sensor"][
            "noise_coefficient"
        ]
        delay = simulation_characteristics_json_data["patient"]["sensor"]["delay"]
        bias_drift_type = simulation_characteristics_json_data["patient"]["sensor"][
            "bias_drift_type"
        ]
        bias_type = simulation_characteristics_json_data["patient"]["sensor"][
            "bias_type"
        ]
        noise_per_sensor = simulation_characteristics_json_data["patient"]["sensor"][
            "noise_per_sensor"
        ]
        noise = simulation_characteristics_json_data["patient"]["sensor"]["noise"]
        bias_factor = simulation_characteristics_json_data["patient"]["sensor"][
            "bias_factor"
        ]
        phi_drift = simulation_characteristics_json_data["patient"]["sensor"][
            "phi_drift"
        ]
        drift_multiplier = simulation_characteristics_json_data["patient"]["sensor"][
            "drift_multiplier"
        ]
        drift_multiplier_start = simulation_characteristics_json_data["patient"][
            "sensor"
        ]["drift_multiplier_start"]
        drift_multiplier_end = simulation_characteristics_json_data["patient"][
            "sensor"
        ]["drift_multiplier_end"]
        noise_max = simulation_characteristics_json_data["patient"]["sensor"][
            "noise_max"
        ]

        # Calculate mard and mbe
        mard = calc_mard(simulation_df)
        mbe = calc_mbe(simulation_df)

    # Parse the bg test condition and analysis type from filename
    bg_test_condition = filename.split(".")[1].replace("bg", "")
    analysis_type = filename.split(".")[3]

    # Calculate risk metrics (using data-science-metrics functions)
    lbgi = metrics.glucose.blood_glucose_risk_index(bg_array=simulation_df["bg"])[0]
    lbgi_rs = metrics.glucose.lbgi_risk_score(lbgi)
    dkai = metrics.insulin.dka_index(simulation_df["iob"], simulation_df["sbr"].iloc[0])
    dkai_rs = metrics.insulin.dka_risk_score(dkai)
    hbgi = metrics.glucose.blood_glucose_risk_index(bg_array=simulation_df["bg"])[1]
    bgri = metrics.glucose.blood_glucose_risk_index(bg_array=simulation_df["bg"])[2]
    percent_lt_54 = metrics.glucose.percent_values_lt_54(bg_array=simulation_df["bg"])

    return [
        filename,
        sim_id,
        virtual_patient_num,
        sensor_num,
        patient_scenario_filename,
        age,
        ylw,
        cir,
        isf,
        sbr,
        starting_bg,
        starting_bg_sensor,
        true_bolus,
        initial_bias,
        bias_norm_factor,
        bias_drift_oscillations,
        bias_drift_range_start,
        bias_drift_range_end,
        noise_coefficient,
        delay,
        bias_drift_type,
        bias_type,
        noise_per_sensor,
        noise,
        bias_factor,
        phi_drift,
        drift_multiplier,
        drift_multiplier_start,
        drift_multiplier_end,
        noise_max,
        mard,
        mbe,
        bg_test_condition,
        analysis_type,
        lbgi,
        lbgi_rs,
        dkai,
        dkai_rs,
        hbgi,
        bgri,
        percent_lt_54,
    ]


# Visualization Functions

# Meta data table code based on legacy make_figures_and_tables.py code

# Generic methods for tables based on bins


def bin_data(bin_breakpoints):
    """

    Parameters
    ----------
    bin_breakpoints: array-like
        Array-like containing Interval objects from which to build the IntervalIndex.

    Returns
    -------
    interval index

    """

    # the bin_breakpoints are the points that are greater than or equal to
    return pd.IntervalIndex.from_breaks(bin_breakpoints, closed="left")


# TODO: THIS FUNCTION NEEDS UPDATED
def get_metadata_tables(demographic_df, fig_path):
    """

    Parameters
    ----------
    demographic_df: dataframe
        dataframe of
    fig_path: str
        path to save figures at

    Returns
    -------

    """

    # Prepare demographic data for tables
    virtual_patient_group = demographic_df.groupby("virtual_patient_num")

    demographic_reduced_df = virtual_patient_group[
        ["age", "ylw", "cir", "isf", "sbr"]
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
        save_fig_path=fig_path,
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
        save_fig_path=fig_path,
    )

    # %% Carb to Insulin Ratio Table
    cir_bin_breakpoints = np.array(
        [
            demographic_reduced_df["cir"].min(),
            5,
            10,
            15,
            20,
            25,
            demographic_reduced_df["cir"].max() + 1,
        ]
    ).astype(int)
    cir_bins = bin_data(cir_bin_breakpoints)

    # make an cir table
    cir_table = pd.DataFrame(index=cir_bins.astype("str"))
    cir_table.index.name = "Carb-to-Insulin-Ratio"

    # cut the data by bin
    demographic_reduced_df["cir_bin"] = np.nan
    demographic_reduced_df["cir_bin"] = pd.cut(demographic_reduced_df["cir"], cir_bins)
    cir_table["Count"] = demographic_reduced_df.groupby("cir_bin")["cir"].count().values

    # add in missing data
    cir_table.loc["Missing", "Count"] = demographic_reduced_df["cir"].isnull().sum()

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
        save_fig_path=fig_path,
    )

    # %% ISF Table
    isf_bin_breakpoints = np.array(
        [
            np.min([demographic_reduced_df["isf"].min(), 5]),
            10,
            25,
            50,
            75,
            100,
            200,
            np.max([400, demographic_reduced_df["isf"].max() + 1]),
        ]
    ).astype(int)
    isf_bins = bin_data(isf_bin_breakpoints)

    # make an isf table
    isf_table = pd.DataFrame(index=isf_bins.astype("str"))
    isf_table.index.name = "Insulin Sensitivity Factor"

    # cut the data by bin
    demographic_reduced_df["isf_bin"] = np.nan
    demographic_reduced_df["isf_bin"] = pd.cut(demographic_reduced_df["isf"], isf_bins)
    isf_table["Count"] = demographic_reduced_df.groupby("isf_bin")["isf"].count().values

    # add in missing data
    isf_table.loc["Missing", "Count"] = demographic_reduced_df["isf"].isnull().sum()

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
        save_fig_path=fig_path,
    )

    # %% Basal Rate (BR) Table
    br_bin_breakpoints = np.append(
        np.arange(0, 1.5, 0.25),
        np.arange(1.5, demographic_reduced_df["sbr"].max() + 0.5, 0.5),
    )
    br_bins = bin_data(br_bin_breakpoints)

    # make an br table
    br_table = pd.DataFrame(index=br_bins.astype("str"))
    br_table.index.name = "Basal Rate"

    # cut the data by bin
    demographic_reduced_df["br_bin"] = np.nan
    demographic_reduced_df["br_bin"] = pd.cut(demographic_reduced_df["sbr"], br_bins)
    br_table["Count"] = demographic_reduced_df.groupby("br_bin")["sbr"].count().values

    # add in missing data
    br_table.loc["Missing", "Count"] = demographic_reduced_df["sbr"].isnull().sum()

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
        save_fig_path=fig_path,
    )

    return


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
    """

    Parameters
    ----------
    table_df: dataframe
        dataframe for making the table from
    image_type: str
        file type ("png","jpg","pdf", etc.) to save image as
    table_name: str
        name to use for the table when saving the figure
    analysis_name: str
        name of the analysis this table is associated with
    cell_height: sized
        height of the cells in the table
    cell_width: sized
        width of the cells in the table
    cell_header_height: sized
        height of the header cells in the table
    view_fig: bool
        whether or not to view the table (opens in browser)
    save_fig: bool
        whether or not to save the table
    save_csv: bool
        whether to save the table contents as a csv
    save_fig_path: str
        file path for where to save the figure


    Returns
    -------

    """
    # TODO: reduce the number of inputs to: df, style_dict, and save_dict

    # Get size (rows, cols) of table
    table_cols = table_df.columns
    n_rows, n_cols = table_df.shape

    # Make table shell
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

    # Set other elements of the table layout
    if len(cell_width) > 1:
        table_width = np.sum(np.asarray(cell_width))
    else:
        table_width = n_cols * cell_width[0]
    table_height = (n_rows + 1.5) * cell_height[0] + cell_header_height[0]
    table_layout = go.Layout(
        margin=dict(l=10, r=10, t=10, b=0), width=table_width, height=table_height
    )

    # Create the figure with the table and table layout
    fig = go.Figure(data=_table, layout=table_layout)

    # Save and/or view the figure
    save_view_fig(
        fig,
        image_type=image_type,
        figure_name=table_name,
        analysis_name=analysis_name,
        view_fig=view_fig,
        save_fig=save_fig,
        save_fig_path=save_fig_path,
        width=table_width,
        height=table_height,
    )

    # Save the table contents as a csv if desired
    file_name = "{}-{}_{}_{}".format(
        analysis_name, table_name, utc_string, code_version
    )

    if save_csv:
        table_df.to_csv(os.path.join(save_fig_path, file_name + ".csv"))

    return


# Spearman Correlation Coefficient table
# TODO: Spearman Correlation Coefficient Table should be QC-ed and
#  possibly add some in line tests

# TODO: this also needs to be updated to include the new set of sensor parameters
def spearman_correlation_table(
    results_df,
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
    """

    Function for create a spearment correlation coefficient table

    Parameters
    ----------
    results_df: dataframe
        dataframe pulling from for data to calculate spearman correlation coefficients from
    image_type: str
        file type ("png","jpg","pdf", etc.) to save image as
    table_name: str
        name to use for the table when saving the figure
    analysis_name: str
        name of the analysis this table is associated with
    cell_header_height: sized
        height of the header cells in the table
    cell_height: sized
        height of the cells in the table
    cell_width: sized
        width of the cells in the table
    view_fig: bool
        whether or not to view the table (opens in browser)
    save_fig: bool
        whether or not to save the table
    save_csv: bool
        whether to save the table contents as a csv
    save_fig_path: str
        file path for where to save the figure

    Returns
    -------

    """

    # Specify the rows and the columns
    rows = [
        "bias_factor",
        "bias_drift_oscillations",
        "bias_drift_range_start",
        "bias_drift_range_end",
        "noise_coefficient",
        "mard",
        "mbe",
    ]

    cols = ["LBGI", "LBGI Risk Score", "DKAI", "DKAI Risk Score"]

    data = {}

    # Create dataframe for table
    for col in cols:
        row_data = []
        for row in rows:
            rho, pval = stats.spearmanr(results_df[row], results_df[col])
            row_data.append("(" + str(round(rho, 3)) + ", " + str(round(pval, 3)) + ")")
        data[col] = row_data

    spearman_correlation_df = pd.DataFrame(data)
    spearman_correlation_df.insert(
        0,
        "",
        [
            "Bias Factor",
            "Bias Drift Oscillations",
            "Bias Drift Range Start",
            "Bias Drift Range End",
            "Noise Coefficient",
            "Mean Absolute Relative Difference",
            "Mean Bias Error",
        ],
    )

    # Make and save table
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


def create_scatter(
    df,
    x_value="cir",
    y_value="LBGI",
    color_value="",
    image_type="png",
    analysis_name="icgm_sensitivity_analysis",
    view_fig=False,
    save_fig=True,
    title="",
    fig_name="",
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
):
    """

    Generic function for creating and saving a plotly express scatterplot.

    Parameters
    ----------
    df: dataframe
        dataframe want to create scatter plot from
    x_value: str
        column from that dataframe want to use for x value
    y_value: str
        column from that dataframe want to use for y value
    color_value: str
        column from that dataframe want to use for color value
    image_type: str
        file type ("png","jpg","pdf", etc.) to save image as
    analysis_name: str
        name of the analysis this scatterplot is associated with (ex. "icgm_sensitivity_analysis"
    view_fig: bool
        whether or not to view the figure
    save_fig: bool
        whether or not to savve the figure
    title: str
        title of the figure
    fig_name: str
        name of the figure
    save_fig_path: str
        path to save the figure at

    Returns
    -------

    """

    # Use this if there is a color value passed in
    if color_value != "":
        df = df.sort_values(by=color_value, ascending=True)
        fig = px.scatter(
            data_frame=df,
            x=x_value,
            y=y_value,
            opacity=0.3,
            color=color_value,
            title=title,
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        fig.update_traces(marker=dict(size=3))

    # Otherwise just create the scatter plot without a color value
    else:
        fig = px.scatter(data_frame=df, x=x_value, y=y_value, opacity=0.3, title=title)
        fig.update_traces(marker=dict(size=3))

    # Save and/or view the figure
    save_view_fig(
        fig,
        image_type,
        fig_name,
        analysis_name,
        view_fig,
        save_fig,
        save_fig_path,
    )

    return


def run_pairwise_comparison(
    results_df,
    baseline_df,
    results_folder_name,
    baseline_folder_name,
    save_fig_path,
    aggregated_tables_filepath,
):
    """

    Parameters
    ----------
    results_df: dataframe
        dataframe for where the results (icgm) data is stored
    baseline_df: dataframe
        dataframe for where the baseline (ideal sensor) data is stored
    results_folder_name: str
        name of the folder where the results are
    save_fig_path: str
        folder of where to save the figures
    aggregated_tables_filepath: str
        file path location of the aggregated tables

    Returns
    -------

    """
    # Create a combined dataframe of results, merging on patient number, analysis type, and bg test condition
    combined_results_df = results_df.merge(
        baseline_df,
        how="left",
        left_on=["virtual_patient_num", "analysis_type", "bg_test_condition"],
        right_on=["virtual_patient_num", "analysis_type", "bg_test_condition"],
        suffixes=("_icgm", "_baseline"),
    )

    # Add in metrics to compare the baseline results to the icgm results
    combined_results_df["lbgi_ratio"] = (
        combined_results_df["lbgi_icgm"] / combined_results_df["lbgi_baseline"]
    )
    combined_results_df["hbgi_ratio"] = (
        combined_results_df["hbgi_icgm"] / combined_results_df["hbgi_baseline"]
    )
    combined_results_df["dkai_ratio"] = (
        combined_results_df["dkai_icgm"] / combined_results_df["dkai_baseline"]
    )
    combined_results_df["bgri_ratio"] = (
        combined_results_df["bgri_icgm"] / combined_results_df["bgri_baseline"]
    )

    combined_results_df["percent_lt_54_ratio"] = (
        combined_results_df["percent_lt_54_icgm"]
        / combined_results_df["percent_lt_54_baseline"]
    )

    combined_results_df["lbgi_difference"] = (
        combined_results_df["lbgi_icgm"] - combined_results_df["lbgi_baseline"]
    )

    combined_results_df["hbgi_difference"] = (
        combined_results_df["hbgi_icgm"] - combined_results_df["hbgi_baseline"]
    )

    combined_results_df["dkai_difference"] = (
        combined_results_df["dkai_icgm"] - combined_results_df["dkai_baseline"]
    )

    combined_results_df["bgri_difference"] = (
        combined_results_df["bgri_icgm"] - combined_results_df["bgri_baseline"]
    )

    combined_results_df["percent_lt_54_difference"] = (
        combined_results_df["percent_lt_54_icgm"]
        - combined_results_df["percent_lt_54_baseline"]
    )

    # Create file name for combined dataframe
    combined_df_file_name = "{}-{}_{}_{}_{}_{}{}".format(
        "pairwise_comparison",
        "combined_df",
        results_folder_name,
        baseline_folder_name,
        utc_string,
        code_version,
        ".csv",
    )

    # Read and save this dataframe to a csv
    combined_results_df.to_csv(
        path_or_buf=os.path.join(aggregated_tables_filepath, combined_df_file_name),
        index=False,
    )

    # Use that combined dataframe to create pairwise comparison figures
    run_pairwise_comparison_figures(combined_results_df, save_fig_path)

    return


def run_pairwise_comparison_figures(combined_results_df, save_fig_path):
    """

    Create a set of figures for viewing and analyzing the results of the pairwise comparison.

    Additional figures can be added to this function as needed for further
    exploratory analysis and final analysis report.

    Parameters
    ----------
    combined_results_df: dataframe
        dataframe of the merged baseline and icgm results that want to use for creating figures

    save_fig_path: str
        path to save figures to

    Returns
    -------

    """

    # Make Paired Comparison Scatter Plot
    create_paired_comparison_scatter_plots(
        combined_results_df,
        fig_path=os.path.join(
            save_fig_path, "distributions-sensor-characteristic-outcome"
        ),
    )

    # Generate cross tab of risk scores
    create_table_paired_risk_score_bins(
        combined_results_df,
        fig_path=os.path.join(save_fig_path, "risk-score-crosstabs"),
    )

    # Generate sensor characteristic scatter plots (distribution of one sensor characteristic by another)
    create_sensor_characteristic_scatters(
        combined_results_df,
        fig_path=os.path.join(save_fig_path, "sensor_characteristic_distributions"),
    )

    # Create scatter plots by analysis level
    create_paired_comparison_by_analysis_level_scatter_plots(
        combined_results_df,
        fig_path=os.path.join(save_fig_path, "scatter_plots_by_analysis_level"),
        analysis_level="analysis_type_label",
    )

    create_paired_comparison_by_analysis_level_scatter_plots(
        combined_results_df,
        fig_path=os.path.join(save_fig_path, "scatter_plots_by_analysis_level"),
        analysis_level="bg_test_condition_label",
    )

    # Add additional figures as needed

    return


def create_frequency_df_for_risk_score_crosstabs(df, metric):
    """

    Helper function to create_table_paired_risk_score_bins figures. Takes in a dataframe
    and gets the count and percentage of scenarios that started in each particular risk
    score bin (for a given metric) and switch to a different particular bin. Returns that
    frequencies/counts dataframe.

    Parameters
    ----------
    df: dataframe
        dataframe to create a frequency/percentage dataframe from
    metric: str
        column of dataframe want breakdown of (frequencies/percentage)

    Returns
    -------
    frenquency_df, a dataframe with frequencies and counts of scenarios that started in
    each particular risk score bin (for a given metric) and switch to a different particular bin

    """

    # Create a starting crosstab
    frequency_df = pd.crosstab(
        df[metric + "_risk_score_string_baseline"],
        df[metric + "_risk_score_string_icgm"],
    )

    # TODO: update this; probably a better way to do this

    # Get percentages (row-wise)
    percentage_df = frequency_df.loc[
        :, frequency_df.columns != metric + "_risk_score_string_baseline"
    ].apply(lambda x: x / x.sum(), axis=1)

    # Format frequency_df to have both count and percentage
    for row in range(len(frequency_df)):
        for col in range(len(frequency_df.columns)):
            frequency_df.iloc[row, col] = (
                str("{:,}".format(frequency_df.iloc[row, col]))
                + " ("
                + str("{:.1%}".format(percentage_df.iloc[row, col]))
                + ")"
            )

    # TODO: add in rows/columns with 0, 0% for cases where there are no scenarios in that particular bucket

    # Reset index
    frequency_df = frequency_df.reset_index()

    return frequency_df


def create_paired_comparison_by_analysis_level_scatter_plots(
    df, fig_path, analysis_level="analysis_type_label"
):
    """

    Create a set of scatter plots showing the distribution of comparison risk metrics by sensor
    characteristic and how this changes across the categories in an analysis level (bg test condition, analysis type)

    Parameters
    ----------
    df: dataframe
        dataframe of combined results
    fig_path: str
        file path for saving figures
    analysis_level: str
        analysis level want to see the scatter plot comparison for

    Returns
    -------

    """

    # Create the directory if it does not exist
    if not os.path.exists(fig_path):
        print("making directory " + fig_path + "...")
        os.makedirs(fig_path)

    sensor_characteristics = ["initial_bias"]

    outcome_metrics = ["lbgi", "hbgi", "dkai"]

    comparison_types = ["difference"]

    df = df.sort_values(by=["bg_test_condition_label_icgm"])

    analysis_level_unique_values = df[analysis_level + "_icgm"].unique()

    # Create a plot for each of the sensor characteristics specified
    for comparison_type in comparison_types:
        for sensor_characteristic in sensor_characteristics:
            n_cols = len(outcome_metrics)
            n_rows = len(analysis_level_unique_values)
            subplot_titles = []

            # Iterate through analysis level values and outcome metrics to get subplot titles
            for analysis_level_value in analysis_level_unique_values:
                for metric in outcome_metrics:
                    if analysis_level == "bg_test_condition":
                        subplot_titles.append(
                            "BG Test Condition "
                            + str(analysis_level_value)
                            + ", "
                            + metric
                            + "_"
                            + comparison_type
                        )
                    else:
                        subplot_titles.append(
                            str(analysis_level_value) + ", " + metric + comparison_type
                        )

            # Make subplots
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=subplot_titles,
                horizontal_spacing=0.1,
            )

            # Iterate through analysis level values and outcome metrics to add plots
            for i, analysis_level_value in enumerate(analysis_level_unique_values):
                for j, metric in enumerate(outcome_metrics):
                    metric_field = metric + "_" + comparison_type

                    reduced_df = df[
                        df[analysis_level + "_icgm"] == analysis_level_value
                    ]

                    fig.add_trace(
                        go.Scatter(
                            x=reduced_df[sensor_characteristic + "_icgm"],
                            y=reduced_df[metric_field],
                            customdata=reduced_df["filename_icgm"],
                            mode="markers",
                            marker=dict(size=4, opacity=0.6),
                            showlegend=False,
                        ),
                        row=i + 1,
                        col=j + 1,
                    )

            x_max_value = np.nanmax(df[sensor_characteristic + "_icgm"])
            x_min_value = np.nanmin(df[sensor_characteristic + "_icgm"])

            y_max_value = max(np.nanmax(df[metric + "_" + comparison_type]) for metric in outcome_metrics)
            y_min_value = min(np.nanmin(df[metric + "_" + comparison_type]) for metric in outcome_metrics)

            analysis_level_dict = {
                "bg_test_condition_label": "BG Test Condition",
                "analysis_type_label": "Analysis Type",
            }

            # Update figure layout and axes
            fig.update_layout(
                title="Outcome Metric "
                + comparison_type
                + ": Baseline vs. iCGM Sensors<br>By "
                + analysis_level_dict[analysis_level],
                legend_title="Risk Scores",
                showlegend=True,
                font_size=6,
            )

            # Standardize x and y axes
            fig.update_yaxes(range=[y_min_value, y_max_value])

            fig.update_xaxes(
                title=sensor_characteristics_dict[sensor_characteristic],
                range=[x_min_value, x_max_value],
            )

            for i in fig["layout"]["annotations"]:
                i["font"] = dict(size=7)

            # Save and/or view figure
            save_view_fig(
                fig,
                image_type="png",
                figure_name="distribution_"
                + comparison_type
                + "_"
                + analysis_level
                + "_sensor_characteristic"
                + "_pairwise_comparison_scatter",
                analysis_name="icgm-sensitivity-analysis",
                view_fig=False,
                save_fig=True,
                save_fig_path=fig_path,
                width=200 * n_cols,
                height=200 * n_rows,
            )

    return


def create_table_paired_risk_score_bins(df, fig_path):
    """

    Creates a set of crosstab tables showing the count and percentage of scenarios
    that started in each particular risk score bin (for a given metric) and switched
    to a different particular bin. Saves those tables

    Parameters
    ----------
    df: dataframe
        dataframe using to create the risk score bin tables
    fig_path: str
        file path to save the tables to


    Returns
    -------

    """

    # Make the save figure directory if it does not yet exist
    if not os.path.exists(fig_path):
        print("making directory " + fig_path + "...")
        os.makedirs(fig_path)

    # Iterate through the different metrics want tables for
    # This list of metrics should be added to, as needed
    for metric in ["dkai", "lbgi"]:

        # Create the count/percentage dataframe from helper function
        frequency_df = create_frequency_df_for_risk_score_crosstabs(df, metric)

        # Rename columns to make clearer
        frequency_df = frequency_df.rename(
            columns={
                metric
                + "_risk_score_string_baseline": metric
                + " Risk Score<br>Rows: Baseline; Columns: iCGM"
            }
        )

        # Make and save the table
        make_table(
            frequency_df,
            table_name=metric + "_paired_risk_score_cross_tab",
            analysis_name="icgm-sensitivity-analysis",
            cell_header_height=[60],
            cell_height=[30],
            cell_width=[250, 125, 125, 125, 125, 125, 125],
            image_type="png",
            view_fig=False,
            save_fig=True,
            save_fig_path=fig_path,
        )

        # Create these tables for each of the bg test conditions
        for bg_test_condition in df["bg_test_condition"].unique():
            # Create a reduced dataframe for a particular test condition
            reduced_df = df[df["bg_test_condition"] == bg_test_condition]

            # Create the count/percentage dataframe from helper function
            frequency_df = create_frequency_df_for_risk_score_crosstabs(
                reduced_df, metric
            )

            # Rename columns to make clearer
            frequency_df = frequency_df.rename(
                columns={
                    metric
                    + " Risk Score String_baseline": metric
                    + " Risk Score: BG Test Condition "
                    + str(bg_test_condition)
                    + "<br>Rows: Baseline; Columns: iCGM"
                }
            )

            # Make and save the table
            make_table(
                frequency_df,
                table_name=metric
                + "_paired_risk_score_cross_tab_bg_test_condition"
                + str(bg_test_condition),
                analysis_name="icgm-sensitivity-analysis",
                cell_header_height=[60],
                cell_height=[30],
                cell_width=[250, 125, 125, 125, 125, 125, 125],
                image_type="png",
                view_fig=False,
                save_fig=True,
                save_fig_path=fig_path,
            )

        # Create these tables for each of the analysis types
        for analysis_type in df["analysis_type_label_icgm"].unique():
            # Create a reduced dataframe for a particular analysis type
            reduced_df = df[df["analysis_type_label_icgm"] == analysis_type]

            # Create the count/percentage dataframe from helper function
            frequency_df = create_frequency_df_for_risk_score_crosstabs(
                reduced_df, metric
            )

            frequency_df = frequency_df.rename(
                columns={
                    metric
                    + " Risk Score String_baseline": metric
                    + " Risk Score: "
                    + str(analysis_type)
                    + "<br>Rows: Baseline; Columns: iCGM"
                }
            )

            # Make and save the table
            make_table(
                frequency_df,
                table_name=metric
                + "_paired_risk_score_cross_tab_"
                + str(analysis_type),
                analysis_name="icgm-sensitivity-analysis",
                cell_header_height=[60],
                cell_height=[30],
                cell_width=[250, 125, 125, 125, 125, 125, 125],
                image_type="png",
                view_fig=False,
                save_fig=True,
                save_fig_path=fig_path,
            )

    return


def create_sensor_characteristic_scatters(df, fig_path):
    """

    Create a set of scatterplots to see the bivariate distribution of the
    sensor characteristics (i.e. how noise varies by across the sensor biases from the fitting)

    Parameters
    ----------
    df: dataframe
        dataframe that contains the sensor characteristics (i.e. combined results dataframe)
    fig_path: str
        file path to save the figures to

    Returns
    -------

    """

    # Create the file path if it does not yet exist
    if not os.path.exists(fig_path):
        print("making directory " + fig_path + "...")
        os.makedirs(fig_path)

    # List of sensor characteristics to see distribution of
    sensor_characteristics = [
        "noise_per_sensor",
        "initial_bias",
        "bias_factor",
        "phi_drift",
        "drift_multiplier_start",
        "drift_multiplier_end",
        "noise_max",
    ]

    # Create a plot for each combination of the sensor characteristics specified
    for i, sensor_characteristic_y in enumerate(sensor_characteristics):
        for j, sensor_characteristic_x in enumerate(sensor_characteristics):
            fig = px.scatter(
                df,
                x=sensor_characteristic_x + "_icgm",
                y=sensor_characteristic_y + "_icgm",
            )

            fig.update_traces(marker=dict(size=2))

            # Save figure
            save_view_fig(
                fig,
                image_type="png",
                figure_name=sensor_characteristic_x
                + "_"
                + sensor_characteristic_y
                + "_sensor_characteristic_distributions",
                analysis_name="icgm-sensitivity-analysis",
                view_fig=False,
                save_fig=True,
                save_fig_path=fig_path,
            )

    return


def create_paired_comparison_scatter_plots(df, fig_path, color_value=""):
    """

    Parameters
    ----------
    df: dataframe
        dataframe to make scatterplots from
    fig_path: str
        file path to save figures at
    color_value: str
        what field to use for color for plots (i.e. for a third dimension)

    Returns
    -------

    """

    # Create directory if it doesn't exist
    if not os.path.exists(fig_path):
        print("making directory " + fig_path + "...")
        os.makedirs(fig_path)

    # Specify which comparison types, outcome metrics, and sensor characteristics to create plots for
    # A plot will be made for each comparison type, outcome metric, sensor characteristic combination
    comparison_types = ["ratio", "difference"]

    outcome_metrics = ["lbgi", "dkai", "hbgi"]

    sensor_characteristics = [
        "mard_icgm",
        "mbe_icgm",
        "initial_bias_icgm",
        "bias_norm_factor_icgm",
        "bias_drift_oscillations_icgm",
        "bias_drift_range_start_icgm",
        "bias_drift_range_end_icgm",
        "noise_coefficient_icgm",
        "delay_icgm",
        "bias_drift_type_icgm",
        "bias_type_icgm",
        "noise_per_sensor_icgm",
        "noise_icgm",
        "bias_factor_icgm",
        "phi_drift_icgm",
        "drift_multiplier_icgm",
        "drift_multiplier_start_icgm",
        "drift_multiplier_end_icgm",
        "noise_max_icgm",
    ]

    # Create plot for each combination of comparison type, outcome metric, sensor characteristic
    for comparison_type, outcome_metric, sensor_characteristic in itertools.product(
        comparison_types, outcome_metrics, sensor_characteristics
    ):
        y_value = outcome_metric + "_" + comparison_type
        create_scatter(
            df=df,
            x_value=sensor_characteristic,
            y_value=y_value,
            color_value=color_value,
            title="Distribution of " + y_value + "<br>By " + sensor_characteristic,
            fig_name="distribution_"
            + outcome_metric
            + "_"
            + comparison_type
            + "_by_"
            + sensor_characteristic,
            save_fig_path=fig_path,
        )

    return


def settings_outside_clinical_bounds(cir, isf, sbr):
    """

    Identifies whether any of the settings are outside clinical bounds (based on medical advisory)

    Parameters
    ----------
    cir: float
        carb to insulin ratio for the particular scenario
    isf: float
        insulin sensitivity factor for the particular scenario
    sbr:float
        scheduled basal rate for the particular scenario

    Returns
    -------
    a boolean for whether any of the settings fall outside of the clinical bounds criteria as defined in the function

    """
    return (
        (float(isf) < 10)
        | (float(isf) > 500)
        | (float(cir) < 2)
        | (float(cir) > 150)
        | (float(sbr) < 0.05)
        | (float(sbr) > 30)
    )


def create_data_frame_for_figures(
    results_path, save_path, results_folder_name, is_baseline=False
):
    """

    Create an aggregate dataframe of the simulation results (for either baseline or iCGM) for
    use in creating figures and analyzig the results. Save those aggregated dataframes as csvs.

    Parameters
    ----------
    results_path: str
        path to where the results want to create dataframe for are
    save_path: str
        file path of outer folder for where to save the csvs of the aggregated dataframes to
    results_folder_name: str
        name of the folder that the results are located in (to be used in title of csvs saved)
    is_baseline: bool
        whether or not this results set is for the baseline (vs. icgm)
        slightly different loading of data for baseline

    Returns
    -------
    dataframe of aggregated results

    """

    # Columns will be loading in data for

    columns = [
        "filename",
        "sim_id",
        "virtual_patient_num",
        "sensor_num",
        "patient_scenario_filename",
        "age",
        "ylw",
        "cir",
        "isf",
        "sbr",
        "starting_bg",
        "starting_bg_sensor",
        "true_bolus",
        "initial_bias",
        "bias_norm_factor",
        "bias_drift_oscillations",
        "bias_drift_range_start",
        "bias_drift_range_end",
        "noise_coefficient",
        "delay",
        "bias_drift_type",
        "bias_type",
        "noise_per_sensor",
        "noise",
        "bias_factor",
        "phi_drift",
        "drift_multiplier",
        "drift_multiplier_start",
        "drift_multiplier_end",
        "noise_max",
        "mard",
        "mbe",
        "bg_test_condition",
        "analysis_type",
        "lbgi",
        "lbgi_risk_score",
        "dkai",
        "dkai_risk_score",
        "hbgi",
        "bgri",
        "percent_lt_54",
    ]

    # Blank list to keep track of scenarios removed because settings are outside of clinical bounds.
    removed_scenarios = []

    # Blank list for adding data to
    data = []

    # Iterate through each of the files
    for i, filename in enumerate(
        sorted(os.listdir(results_path))
    ):  # [0:100])): #(for testing)
        # Identify file is simulation file
        if filename.endswith(".tsv"):

            print(i, filename)

            # Read in that simulation data to a dataframe
            simulation_df = pd.read_csv(os.path.join(results_path, filename), sep="\t")

            # Check that the first two bg values are equal
            assert (
                simulation_df.loc[0]["bg"] == simulation_df.loc[1]["bg"]
            ), "First two BG values of simulation are not equal"

            # Find and read in the corresponding json data
            f = open(os.path.join(results_path, filename.replace(".tsv", ".json")), "r")
            simulation_characteristics_json_data = json.loads(f.read())

            # Get the scenario settings characteristics so can check whether outside clinical bounds
            cir = simulation_characteristics_json_data["patient"]["config"][
                "carb_ratio_schedule"
            ]["schedule"][0]["setting"].replace(" g", "")

            isf = simulation_characteristics_json_data["patient"]["config"][
                "insulin_sensitivity_schedule"
            ]["schedule"][0]["setting"].replace(" m", "")

            sbr = simulation_characteristics_json_data["patient"]["config"][
                "basal_schedule"
            ]["schedule"][0]["setting"].replace(" U", "")

            # If any of the settings are outside clinical bounds, add to removed scnearios list and do not load
            # into aggregated dataframe
            if settings_outside_clinical_bounds(cir, isf, sbr):
                print(filename + " has settings outside clinical bounds.")
                removed_scenarios.append([filename, cir, isf, sbr])

            # Otherwise add that data to data list
            else:

                # Get a row of data for that particular simulation and scenario characteristics
                data_row = get_data(
                    filename,
                    simulation_df,
                    simulation_characteristics_json_data,
                    baseline=is_baseline,
                )

                # Confirm that the length of the returned data matches the number of columns for
                # ultimate aggregate datafrrame
                assert len(data_row) == len(
                    columns
                ), "length of returned data does not match number of columns"

                data.append(data_row)

        # Create and save dataframe of removed scenarios
        removed_scenarios_df = pd.DataFrame(
            removed_scenarios, columns=["filename", "cir", "isf", "sbr"]
        )
        removed_scenarios_filename = "{}_{}_{}_{}".format(
            "removed_scenarios_df", results_folder_name, utc_string, code_version
        )

        removed_scenarios_df.to_csv(
            path_or_buf=os.path.join(save_path, removed_scenarios_filename + ".csv"),
            index=False,
        )

    # Create the results dataframe
    results_df = pd.DataFrame(data, columns=columns)

    # Clean up the results dataframe
    results_df = clean_up_results_df(results_df)

    # Save the results dataframe to a csv

    results_df_filename = "{}_{}_{}_{}".format(
        "results_df", results_folder_name, utc_string, code_version
    )

    results_df.to_csv(
        path_or_buf=os.path.join(save_path, results_df_filename + ".csv"),
        index=False,
    )

    return results_df


def clean_up_results_df(results_df):
    """

    Parameters
    ----------
    results_df: dataframe
        dataframe to clean up

    Returns
    -------

    """
    # Make age and ylw columns numeric
    results_df[["age", "ylw"]] = results_df[["age", "ylw"]].apply(pd.to_numeric)

    # TODO: these additional columns should maybe just be replaced by doing cleanup within the figure code itself

    # Add additional columns for help with cleaning up figures
    results_df["analysis_type_label"] = results_df["analysis_type"].replace(
        analysis_type_labels
    )
    results_df["bg_test_condition_label"] = results_df["bg_test_condition"].replace(
        analysis_type_labels
    )
    results_df["dkai_risk_score_string"] = results_df["dkai_risk_score"].replace(
        score_dict
    )
    results_df["lbgi_risk_score_string"] = results_df["lbgi_risk_score"].replace(
        score_dict
    )

    return results_df


# Dictionaries for use in visualization functions
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

level_of_analysis_dict = {
    "all": "All Analyses",
    "analysis_type": "Analysis Type",
    "bg_test_condition": "BG Test Condition",
}

# TODO: use the sensor characteristics dictionary to clean up titles and axis names for final report figures
sensor_characteristics_dict = {
    "noise_per_sensor": "Noise Per Sensor",
    "initial_bias": "Initial Bias",
    "bias_factor": "Bias Factor",
    "phi_drift": "Phi Drift",
    "drift_multiplier_start": "Drift Multiplier Start",
    "drift_multiplier_end": "Drift Multiplier End",
    "noise_max": "Noise Max",
}

if __name__ == "__main__":

    # To create the iCGM sensitivity analysis figures, follow these steps:
    # 1. Get baseline and results data from Google Drive and add the data folders to .../data/raw
    # 2. Optional: add "-nogit" to the end of those files (this will add "-nogit" to figure directories, as well)
    # or add to the gitignore.
    # 3. Set "data_already_loaded" to False and run this code.
    # The aggregate tables will be written to .../data/processed and can be used for any subsequent figure creation.
    #
    # For all subsequent runs of this same data:
    # 1. Set data_already_loaded to True
    # 2. Update icgm_data_filename and combined_data_filename to be the names of the aggregated
    # results dataframes that have already been created.


    # TODO: automatically grab the code version to add to the figures generated
    code_version = "v0-1-0"
    utc_string = dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%m-%S")

    # Specify this parameter based on whether want to load the data and run the figures
    # or just run the figures
    data_already_loaded = True

    # Specify the iCGM data filepath
    icgm_folder_name = "icgm-sensitivity-analysis-results-2020-11-02-nogit"
    results_files_path = os.path.join("..", "..", "data", "raw", icgm_folder_name)

    # Specify the Baseline data filepath
    ideal_sensor_folder_name = "icgm-sensitivity-analysis-results-2020-11-05-nogit"
    baseline_files_path = os.path.join(
        "..", "..", "data", "raw", ideal_sensor_folder_name
    )

    # Save folder path
    save_folder_name = "{}_{}".format(
        icgm_folder_name, ideal_sensor_folder_name
    )



    # Specify where to save the figures to
    results_save_fig_path = os.path.join(
        "..",
        "..",
        "reports",
        "figures",
        "icgm-sensitivity-paired-comparison-figures",
        save_folder_name,
    )

    # Create the directory if it does not exist
    if not os.path.exists(results_save_fig_path):
        print("making directory " + results_save_fig_path + "...")
        os.makedirs(results_save_fig_path)

    # Specify where to save the metadata tables to
    metadata_save_path = os.path.join(results_save_fig_path, "metadata_tables")

    # Create the directory if it does not exist
    if not os.path.exists(metadata_save_path):
        print("making directory " + metadata_save_path + "...")
        os.makedirs(metadata_save_path)

    # Specify where to save the process data csv files to
    processed_data_filepath = os.path.join(
        "..",
        "..",
        "data",
        "processed",
        "icgm-sensitivity-analysis-aggregate-tables-nogit",
        save_folder_name,
    )

    # Create the directory if it does not exist
    if not os.path.exists(processed_data_filepath):
        print("making directory " + processed_data_filepath + "...")
        os.makedirs(processed_data_filepath)

    if not data_already_loaded:
        icgm_results_df = create_data_frame_for_figures(
            is_baseline=False,
            results_path=results_files_path,
            save_path=processed_data_filepath,
            results_folder_name=icgm_folder_name,
        )

        baseline_sensor_df = create_data_frame_for_figures(
            is_baseline=True,
            results_path=baseline_files_path,
            save_path=processed_data_filepath,
            results_folder_name=ideal_sensor_folder_name,
        )

        run_pairwise_comparison(
            results_df=icgm_results_df,
            baseline_df=baseline_sensor_df,
            results_folder_name=icgm_folder_name,
            baseline_folder_name=ideal_sensor_folder_name,
            save_fig_path=results_save_fig_path,
            aggregated_tables_filepath=processed_data_filepath,
        )

        get_metadata_tables(icgm_results_df, fig_path=metadata_save_path)

    else:  # Just create the figures (don't load the datatables)

        # Create the pairwise comparison dataframe
        combined_data_filename = "pairwise_comparison-combined_df_icgm-sensitivity-analysis-results-2020-11-02-nogit_" \
                                 "icgm-sensitivity-analysis-results-2020-11-05-nogit_2020-11-23-03-11-30_v0-1-0.csv"
        combined_data_filepath = os.path.join(
            processed_data_filepath, combined_data_filename
        )

        combined_df = pd.read_csv(combined_data_filepath)
        run_pairwise_comparison_figures(combined_df, results_save_fig_path)

        # Create metadata tables
        icgm_data_filename = "results_df_icgm-sensitivity-analysis-results-2020-11-05-nogit_2020-11-23-03-11-30_v0-1-0.csv"

        icgm_results_df = pd.read_csv(
            os.path.join(processed_data_filepath, icgm_data_filename)
        )

        get_metadata_tables(icgm_results_df, fig_path=metadata_save_path)
