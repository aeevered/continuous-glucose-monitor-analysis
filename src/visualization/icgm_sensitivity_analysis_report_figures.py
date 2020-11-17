# %% REQUIRED LIBRARIES
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
import itertools
from save_view_fig import save_view_fig
import json
from scipy import stats
import tidepool_data_science_metrics as metrics
from risk_scenario_figures_plotly import create_simulation_figure_plotly
from risk_scenario_figures_shared_functions import data_loading_and_preparation

# from tidepool_data_science_models.models.icgm_sensor_generator_functions import (
#     calc_mard,
#     preprocess_data,
#     calc_mbe,
#     calc_icgm_sc_table,
#     calc_icgm_special_controls_loss,
# )

utc_string = dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%m-%S")

# Calculate MBE and MARD (https://github.com/tidepool-org/icgm-sensitivity-analysis/blob/jameno/analysis-tables/src/simulator_functions.py)
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

    df["withinMeasRange"] = (sensor_bg_values >= sensor_min) & (
        sensor_bg_values <= sensor_max
    )

    return df


def calc_mbe(df):

    # default icgm and ysi ranges [40, 400] and [0, 900]

    df = add_error_fields(df)
    return np.mean(df.loc[df["withinMeasRange"], "icgmError"])


def calc_mard(df):
    """ Mean Absolute Relative Deviation (MARD)
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5375072/
    """

    df = add_error_fields(df)

    abs_relative_difference_in_measurement_range = df.loc[
        df["withinMeasRange"], "absRelDiff"
    ]
    return np.mean(abs_relative_difference_in_measurement_range)


# Parse out simulation id
def get_sim_id(patient_characteristics_df, filename):
    sensor_num = (
        filename.split("/")[-1]
        .split(".")[2]
        .replace("s", "")
        .replace("Senor", "Sensor")
    )
    vp_id = (
        patient_characteristics_df["patient_scenario_filename"]
        .iloc[0]
        .split("/")[-1]
        .split(".")[0]
        .replace("train_", "")
    )
    bg_test_condition = filename.split(".")[1]
    analysis_type = filename.split(".")[3]
    sim_id = (
        "vp"
        + str(vp_id)
        + ".bg"
        + ".s"
        + str(sensor_num)
        + "."
        + str(bg_test_condition)
        + "."
        + analysis_type
    )
    return sim_id

def get_data(filename, simulation_df, simulation_characteristics_json_data, baseline=False):
    sim_id = simulation_characteristics_json_data["sim_id"]
    virtual_patient_num = simulation_characteristics_json_data["sim_id"].split(".")[0]
    sensor_num = filename.split(".")[2]
    patient_scenario_filename = (
        filename.split(".")[0]
    )
    age = simulation_characteristics_json_data["controller"]["config"]["age"]
    ylw = simulation_characteristics_json_data["controller"]["config"]["ylw"]
    cir = simulation_characteristics_json_data["patient"]["config"][
        "carb_ratio_schedule"
    ]["schedule"][0]["setting"]
    isf = simulation_characteristics_json_data["patient"]["config"][
        "insulin_sensitivity_schedule"
    ]["schedule"][0]["setting"]
    sbr = simulation_characteristics_json_data["patient"]["config"]["basal_schedule"][
        "schedule"
    ][0]["setting"]

    starting_bg = simulation_df["bg"].iloc[0]
    starting_bg_sensor = simulation_df["bg_sensor"].iloc[0]
    true_bolus = simulation_df["true_bolus"].iloc[1]

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

        mard = calc_mard(simulation_df)
        mbe = calc_mbe(simulation_df)

    bg_test_condition = filename.split(".")[1].replace("bg", "")
    analysis_type = filename.split(".")[3]

    LBGI = metrics.glucose.blood_glucose_risk_index(bg_array=simulation_df["bg"])[0]
    LBGI_RS = metrics.glucose.lbgi_risk_score(LBGI)
    DKAI = metrics.insulin.dka_index(simulation_df["iob"], simulation_df["sbr"].iloc[0])
    DKAI_RS = metrics.insulin.dka_risk_score(DKAI)
    HBGI = metrics.glucose.blood_glucose_risk_index(bg_array=simulation_df["bg"])[1]
    BGRI = metrics.glucose.blood_glucose_risk_index(bg_array=simulation_df["bg"])[2]
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
        LBGI,
        LBGI_RS,
        DKAI,
        DKAI_RS,
        HBGI,
        BGRI,
        percent_lt_54,
    ]


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


def get_metadata_tables(demographic_df, fig_path):
    # %% prepare demographic data for tables

    print(demographic_df.columns)
    virtual_patient_group = demographic_df[["virtual_patient_num","age", "ylw", "CIR", "ISF", "SBR"]].groupby("virtual_patient_num")
    print(virtual_patient_group.columns)
    demographic_reduced_df = virtual_patient_group[
        ["age", "ylw", "CIR", "ISF", "SBR"]
    ].median()
    print(demographic_reduced_df.columns)

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
        save_fig_path=fig_path,
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
        save_fig_path=fig_path,
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
        save_fig_path=fig_path,
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

    # print(table_height, table_width)

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

    """
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
    """
    file_name = "{}-{}_{}_{}".format(
        analysis_name, table_name, utc_string, code_version
    )

    if save_csv:
        table_df.to_csv(os.path.join(save_fig_path, file_name + ".csv"))

    return

########## Spearman Correlation Coefficient Table #################
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
    color_value="",
    image_type="png",
    analysis_name="icgm_sensitivity_analysis",
    view_fig=False,
    save_fig=True,
    title="",
    fig_name="",
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
):

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
        )  # , color_continuous_scale=px.colors.diverging.RdYlGn)
        fig.update_traces(marker=dict(size=3))
    else:
        fig = px.scatter(data_frame=df, x=x_value, y=y_value, opacity=0.3, title=title)

        fig.update_traces(marker=dict(size=3))

    save_view_fig(
        fig, image_type, fig_name, analysis_name, view_fig, save_fig, save_fig_path,
    )
    return

def run_pairwise_comparison(results_df, baseline_df, save_fig_folder_name):
    # Add ratio to each row
    # Need to look up for each row into the baseline_df by virtual patient and by
    fig_path = os.path.join(
        "..",
        "..",
        "reports",
        "figures",
        "icgm-sensitivity-paired-comparison-figures",
        save_fig_folder_name,
    )

    if not os.path.exists(fig_path):
        print("making directory " + fig_path + "...")
        os.makedirs(fig_path)

    combined_df = results_df.merge(
        baseline_df,
        how="left",
        left_on=["virtual_patient_num", "analysis_type", "bg_test_condition"],
        right_on=["virtual_patient_num", "analysis_type", "bg_test_condition"],
        suffixes=("_icgm", "_baseline"),
    )

    combined_df["LBGI Ratio"] = combined_df["LBGI_icgm"] / combined_df["LBGI_baseline"]
    combined_df["HBGI Ratio"] = combined_df["HBGI_icgm"] / combined_df["HBGI_baseline"]
    combined_df["DKAI Ratio"] = combined_df["DKAI_icgm"] / combined_df["DKAI_baseline"]
    combined_df["BGRI Ratio"] = combined_df["BGRI_icgm"] / combined_df["BGRI_baseline"]
    combined_df["Percent <54 Ratio"] = (
        combined_df["percent_lt_54_icgm"] / combined_df["percent_lt_54_baseline"]
    )

    combined_df["LBGI Percent Change"] = (
        (combined_df["LBGI_icgm"] - combined_df["LBGI_baseline"]) * 100
    ) / combined_df["LBGI_baseline"]
    combined_df["HBGI Percent Change"] = (
        (combined_df["HBGI_icgm"] - combined_df["HBGI_baseline"]) * 100
    ) / combined_df["HBGI_baseline"]
    combined_df["DKAI Percent Change"] = (
        (combined_df["DKAI_icgm"] - combined_df["DKAI_baseline"]) * 100
    ) / combined_df["DKAI_baseline"]
    combined_df["BGRI Percent Change"] = (
        (combined_df["BGRI_icgm"] - combined_df["BGRI_baseline"]) * 100
    ) / combined_df["BGRI_baseline"]
    combined_df["Percent <54 Percent Change"] = (
        (combined_df["percent_lt_54_icgm"] - combined_df["percent_lt_54_baseline"])
        * 100
    ) / combined_df["percent_lt_54_baseline"]

    combined_df["LBGI Difference"] = (
        combined_df["LBGI_icgm"] - combined_df["LBGI_baseline"]
    )
    combined_df["HBGI Difference"] = (
        combined_df["HBGI_icgm"] - combined_df["HBGI_baseline"]
    )
    combined_df["DKAI Difference"] = (
        combined_df["DKAI_icgm"] - combined_df["DKAI_baseline"]
    )
    combined_df["BGRI Difference"] = (
        combined_df["BGRI_icgm"] - combined_df["BGRI_baseline"]
    )
    combined_df["Percent <54 Difference"] = (
        combined_df["percent_lt_54_icgm"] - combined_df["percent_lt_54_baseline"]
    )

    combined_df.to_csv(
        path_or_buf=os.path.join(
            fig_path, "pairwise_comparison_combined_df_" + save_fig_folder_name +".csv"
        ),
        index=False,
    )

    run_pairwise_comparison_figures(save_fig_folder_name)

    return


def run_pairwise_comparison_figures(save_fig_folder_name):

    fig_path = os.path.join(
        "..",
        "..",
        "reports",
        "figures",
        "icgm-sensitivity-paired-comparison-figures",
        save_fig_folder_name,
    )
    combined_df = pd.read_csv(
        os.path.join(
            "..",
            "..",
            "reports",
            "figures",
            "icgm-sensitivity-paired-comparison-figures",
            save_fig_folder_name,
            "pairwise_comparison_combined_df_" + save_fig_folder_name + ".csv",
        )
    )

    # Make Paired Comparison Scatter Plot
    create_paired_comparison_scatter_plots(
        combined_df,
        fig_path=os.path.join(fig_path, "distributions-sensor-characteristic-outcome"),
    )

    # Generate crosstab of risk scores
    create_table_paired_risk_score_bins(
       combined_df, fig_path=os.path.join(fig_path, "risk-score-crosstabs")
    )

    create_sensor_characteristic_scatters(
        combined_df, fig_path=os.path.join(fig_path, "sensor_characteristic_distributions")
    )

    return


def create_table_paired_risk_score_bins(df, fig_path):
    if not os.path.exists(fig_path):
        print("making directory " + fig_path + "...")
        os.makedirs(fig_path)

    for metric in ["DKAI", "LBGI"]:
        frequency_df = pd.crosstab(
            df[metric + " Risk Score String_baseline"],
            df[metric + " Risk Score String_icgm"],
        )  # .reset_index()

        # TODO: update this; not pythonic way to do
        percentage_df = frequency_df.loc[
            :, frequency_df.columns != metric + " Risk Score String_baseline"
        ].apply(lambda x: x / x.sum(), axis=1)
        for row in range(len(frequency_df)):
            for col in range(len(frequency_df.columns)):
                frequency_df.iloc[row, col] = (
                    str("{:,}".format(frequency_df.iloc[row, col]))
                    + " ("
                    + str("{:.1%}".format(percentage_df.iloc[row, col]))
                    + ")"
                )

        frequency_df = frequency_df.reset_index()

        frequency_df = frequency_df.rename(
            columns={
                metric
                + " Risk Score String_baseline": metric
                + " Risk Score<br>Rows: Baseline; Columns: iCGM"
            }
        )

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

        for bg_test_condition in df["bg_test_condition"].unique():
            reduced_df = df[df["bg_test_condition"] == bg_test_condition]
            frequency_df = pd.crosstab(
                reduced_df[metric + " Risk Score String_baseline"],
                reduced_df[metric + " Risk Score String_icgm"],
            )  # .reset_index()

            # TODO: update this; not pythonic way to do
            percentage_df = frequency_df.loc[
                :, frequency_df.columns != metric + " Risk Score String_baseline"
            ].apply(lambda x: x / x.sum(), axis=1)
            for row in range(len(frequency_df)):
                for col in range(len(frequency_df.columns)):
                    frequency_df.iloc[row, col] = (
                        str("{:,}".format(frequency_df.iloc[row, col]))
                        + " ("
                        + str("{:.1%}".format(percentage_df.iloc[row, col]))
                        + ")"
                    )

            frequency_df = frequency_df.reset_index()

            frequency_df = frequency_df.rename(
                columns={
                    metric
                    + " Risk Score String_baseline": metric
                    + " Risk Score: BG Test Condition "
                    + str(bg_test_condition)
                    + "<br>Rows: Baseline; Columns: iCGM"
                }
            )

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

        for analysis_type in df["analysis_type_label_icgm"].unique():
            reduced_df = df[df["analysis_type_label_icgm"] == analysis_type]
            frequency_df = pd.crosstab(
                reduced_df[metric + " Risk Score String_baseline"],
                reduced_df[metric + " Risk Score String_icgm"],
            )  # .reset_index()

            # TODO: update this; not pythonic way to do
            percentage_df = frequency_df.loc[
                :, frequency_df.columns != metric + " Risk Score String_baseline"
            ].apply(lambda x: x / x.sum(), axis=1)
            for row in range(len(frequency_df)):
                for col in range(len(frequency_df.columns)):
                    frequency_df.iloc[row, col] = (
                        str("{:,}".format(frequency_df.iloc[row, col]))
                        + " ("
                        + str("{:.1%}".format(percentage_df.iloc[row, col]))
                        + ")"
                    )

            frequency_df = frequency_df.reset_index()

            frequency_df = frequency_df.rename(
                columns={
                    metric
                    + " Risk Score String_baseline": metric
                    + " Risk Score: "
                    + str(analysis_type)
                    + "<br>Rows: Baseline; Columns: iCGM"
                }
            )

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
                view_fig=True,
                save_fig=True,
                save_fig_path=fig_path,
            )

    return

def create_sensor_characteristic_scatters(df, fig_path):
    if not os.path.exists(fig_path):
        print("making directory " + fig_path + "...")
        os.makedirs(fig_path)

    sensor_characteristics = [
        "noise_per_sensor",
        "initial_bias",
        "bias_factor",
        "phi_drift",
        "drift_multiplier_start",
        "drift_multiplier_end",
        "noise_max",
    ]

    sensor_characteristics_dict = {
        "noise_per_sensor": "Noise Per Sensor",
        "initial_bias": "Initial Bias",
        "bias_factor": "Bias Factor",
        "phi_drift": "Phi Drift",
        "drift_multiplier_start": "Drift Multiplier Start",
        "drift_multiplier_end": "Drift Multiplier End",
        "noise_max": "Noise Max",
    }

    # Create a plot for each of the sensor characteristics specified
    for i, sensor_characteristic_y in enumerate(sensor_characteristics):
        for j, sensor_characteristic_x in enumerate(sensor_characteristics):

            fig = px.scatter(df, x=sensor_characteristic_x + "_icgm", y=sensor_characteristic_y + "_icgm")
            fig.show()

            save_view_fig(
                fig,
                image_type="png",
                figure_name=sensor_characteristic_x + "_" + sensor_characteristic_y + "_sensor_characteristic_distributions",
                analysis_name="icgm-sensitivity-analysis",
                view_fig=False,
                save_fig=True,
                save_fig_path=fig_path
            )

    return

def create_paired_comparison_scatter_plots(combined_df, fig_path, color_value=""):
    if not os.path.exists(fig_path):
        print("making directory " + fig_path + "...")
        os.makedirs(fig_path)

    comparison_types = [" Ratio", " Percent Change", " Difference"]

    outcome_metrics = ["LBGI", "DKAI", "HBGI"]  # "Percent <54"]

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

    for comparison_type, outcome_metric, sensor_characteristic in itertools.product(
        comparison_types, outcome_metrics, sensor_characteristics
    ):
        create_scatter(
            df=combined_df,
            x_value=sensor_characteristic,
            y_value=outcome_metric + comparison_type,
            color_value=color_value,
            title="Distribution of "
            + outcome_metric
            + comparison_type
            + "<br>By "
            + sensor_characteristic,
            fig_name="distribution_"
            + outcome_metric
            + "_"
            + comparison_type
            + "_by_"
            + sensor_characteristic,
            save_fig_path=fig_path,
        )

    return

def visualize_individual_sim_result(df, icgm_path, baseline_path, save_fig_path, save_fig_folder_name, animation_filenames = []):
    # animation_filenames = df.loc[
    #     ((df["HBGI Difference"] > 20) | (df["HBGI Difference"] < -20)), "filename_icgm"
    # ]
    # df.loc[((df['DKAI Difference'] > 5) | (df['LBGI Difference'] > 5)), 'filename_icgm']

    # For testing
    # animation_filenames = ["vp12.bg9.s1.correction_bolus.csv"]
    # print(len(animation_filenames))

    for i, filename in enumerate(animation_filenames[0:10]):

        print(i, filename)

        baseline_filename = df.loc[
            df["filename_icgm"] == filename, "filename_baseline"
        ].iloc[0]

        icgm_DKAI_RS = df.loc[
            df["filename_icgm"] == filename, "DKAI Risk Score_icgm"
        ].iloc[0]
        icgm_LBGI_RS = df.loc[
            df["filename_icgm"] == filename, "LBGI Risk Score_icgm"
        ].iloc[0]

        baseline_DKAI_RS = df.loc[
            df["filename_icgm"] == filename, "DKAI Risk Score_baseline"
        ].iloc[0]
        baseline_LBGI_RS = df.loc[
            df["filename_icgm"] == filename, "LBGI Risk Score_baseline"
        ].iloc[0]

        mard_icgm = df.loc[
            df["filename_icgm"] == filename, "mard_icgm"
        ].iloc[0]

        initial_bias_icgm = df.loc[
            df["filename_icgm"] == filename, "initial_bias_icgm"
        ].iloc[0]

        icgm_simulation_df = data_loading_and_preparation(
            os.path.join(icgm_path, filename)
        )

        baseline_simulation_df = data_loading_and_preparation(
            os.path.join(baseline_path, baseline_filename)
        )

        # List of dictionaries
        traces = [
            {0: ["bg", "bg_sensor"], 1: ["sbr", "temp_basal_sbr_if_nan"]},
            {2: ["bg", "bg_sensor"], 3: ["sbr", "temp_basal_sbr_if_nan"]},
        ]

        print(baseline_simulation_df.columns)
        print(icgm_simulation_df.columns)

        max_basal = (
            max(
                np.nanmax(baseline_simulation_df["sbr"]),
                np.nanmax(icgm_simulation_df["sbr"]),
                np.nanmax(baseline_simulation_df["temp_basal"]),
                np.nanmax(icgm_simulation_df["temp_basal"]),
            )
            + 0.5
        )
        max_bg = (
            max(
                np.nanmax(baseline_simulation_df["bg"]),
                np.nanmax(icgm_simulation_df["bg"]),
                np.nanmax(baseline_simulation_df["bg_sensor"]),
                np.nanmax(icgm_simulation_df["bg_sensor"]),
            )
            + 20
        )
        min_bg = (
            min(
                np.nanmin(baseline_simulation_df["bg"]),
                np.nanmin(icgm_simulation_df["bg"]),
                np.nanmin(baseline_simulation_df["bg_sensor"]),
                np.nanmin(icgm_simulation_df["bg_sensor"]),
            )
            - 10
        )


        create_simulation_figure_plotly(
            files_need_loaded=False,
            data_frames=[icgm_simulation_df, baseline_simulation_df],
            file_location=os.path.join("..", "..", "data", "processed"),
            file_names=[filename, baseline_filename],
            traces=traces,
            show_legend=False,
            subplots=4,
            time_range=(0, 8),
            subtitle="",
            main_title="iCGM: DKAI RS "
             + str(icgm_DKAI_RS)
             + ", LBGI RS "
             + str(icgm_LBGI_RS)
            + ",  MARD: "
            + str(int(mard_icgm))
           + ",  Initial Bias: "
           + str(int(initial_bias_icgm))
            + " ; Baseline: DKAI RS "
            + str(int(baseline_DKAI_RS))
            + ", LBGI RS "
            + str(int(baseline_LBGI_RS))
            + "<br>"
            + filename,
            subplot_titles=[
                "BG Values (iCGM)",
                "Scheduled Basal Rate and Loop Decisions (iCGM)",
                "BG Values (Baseline)",
                "Scheduled Basal Rate and Loop Decisions (Baseline)",
            ],
            save_fig_path=os.path.join(
                save_fig_path,
                "example_simulations",
                save_fig_folder_name,
            ),
            figure_name= filename,
            analysis_name="icgm_sensitivity_analysis",
            animate=False,
            custom_axes_ranges=[
                (min(50, min_bg), max(260, max_bg)),
                (0, max_basal),
                (min(50, min_bg), max(260, max_bg)),
                (0, max_basal),
            ],
            custom_tick_marks=[
                [54, 70, 140, 180, 250, 300, 350, 400],
                np.arange(0, max_basal, 0.5),
                [54, 70, 140, 180, 250],
                +np.arange(0, max_basal, 0.5),
            ],
        )

    return


def create_sensor_characteristics_table(df, fig_path):
    columns = [
        "sensor_num_icgm",
        "initial_bias_icgm",
        "bias_factor_icgm",
        "bias_drift_oscillations_icgm",
        "bias_drift_range_start_icgm",
        "bias_drift_range_end_icgm",
        "noise_coefficient_icgm",
    ]
    sensor_characteristics_df = df[columns].drop_duplicates()

    sensor_characteristics_df = sensor_characteristics_df.sort_values(
        by=["sensor_num_icgm"]
    )

    sensor_characteristics_df = sensor_characteristics_df.rename(
        columns={
            "sensor_num_icgm": "iCGM Sensor Number",
            "initial_bias_icgm": "Initial Bias",
            "bias_factor_icgm": "Bias Factor",
            "bias_drift_oscillations_icgm": "Bias Factor Oscillations",
            "bias_drift_range_start_icgm": "Bias Drift Range Start",
            "bias_drift_range_end_icgm": "Bias Drift Range End",
            "noise_coefficient_icgm": "Noise Coefficient",
        }
    )

    print(sensor_characteristics_df)

    return


def settings_outside_clinical_bounds(cir, isf, sbr):

    return ((float(isf) < 10) | (float(isf) > 500) | (float(cir) < 2) | (float(cir) > 150) | (float(sbr) < 0.05) | (
                float(sbr) > 30))


def create_data_frame_for_figures(
    results_path,
    save_path,
    results_folder_name,
    is_baseline=False
):

    removed_scenarios = []

    for i, filename in enumerate(sorted(os.listdir(results_path))): #[0:100])):
        if filename.endswith(".tsv"):

            print(i, filename)
            simulation_df = pd.read_csv(
                os.path.join(results_path, filename), sep="\t"
            )

            #Check that the first two bg values are equal
            assert (simulation_df.loc[0]["bg"] == simulation_df.loc[1][
                "bg"]), "First two BG values of simulation are not equal"

            f = open(
                os.path.join(results_path, filename.replace(".tsv", ".json")), "r"
            )
            simulation_characteristics_json_data = json.loads(f.read())

            vp_id = filename.split(".")[0].replace("vp", "")

            cir = simulation_characteristics_json_data["patient"]["config"]["carb_ratio_schedule"]["schedule"][0]["setting"].replace(" g", "")
            isf = simulation_characteristics_json_data["patient"]["config"]["insulin_sensitivity_schedule"]["schedule"][0]["setting"].replace(
                " m", "")
            sbr = simulation_characteristics_json_data["patient"]["config"]["basal_schedule"]["schedule"][0]["setting"].replace(" U", "")

            # Add in the data
            #if vp_id not in vp_outside_clinical_bounds:
            if settings_outside_clinical_bounds(cir, isf, sbr):
                print(filename + " has settings outside clinical bounds.")
                removed_scenarios.append([filename, cir, isf, sbr])

            else:
                data.append(
                    get_data(
                        filename,
                        simulation_df,
                        simulation_characteristics_json_data,
                        baseline=is_baseline,
                    )
                )

        removed_scenarios_df = pd.DataFrame(removed_scenarios, columns=["filename", "cir", "isf", "sbr"])
        removed_scenarios_df.to_csv(
            path_or_buf=os.path.join(save_path, results_folder_name + "_removed_scenarios_df.csv"),
            index=False,
        )

    columns = [
        "filename",
        "sim_id",
        "virtual_patient_num",
        "sensor_num",
        "patient_scenario_filename",
        "age",
        "ylw",
        "CIR",
        "ISF",
        "SBR",
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
        "LBGI",
        "LBGI Risk Score",
        "DKAI",
        "DKAI Risk Score",
        "HBGI",
        "BGRI",
        "percent_lt_54",
    ]

    results_df = pd.DataFrame(data, columns=columns)

    results_df = clean_up_results_df(results_df)

    results_df.to_csv(
        path_or_buf=os.path.join(save_path, results_folder_name + "_results_df.csv"),
        index=False,
    )

    return results_df

def clean_up_results_df(results_df):

    results_df[["age", "ylw"]] = results_df[["age", "ylw"]].apply(pd.to_numeric)

    # rename the analysis types
    results_df.replace({"tempBasal": "Temp Basal Analysis"}, inplace=True)
    results_df.replace({"correctionBolus": "Correction Bolus Analysis"}, inplace=True)

    results_df["analysis_type_label"] = results_df["analysis_type"].replace(
        analysis_type_labels
    )
    results_df["bg_test_condition_label"] = results_df["bg_test_condition"].replace(
        analysis_type_labels
    )
    results_df["DKAI Risk Score String"] = results_df["DKAI Risk Score"].replace(
        score_dict
    )
    results_df["LBGI Risk Score String"] = results_df["LBGI Risk Score"].replace(
        score_dict
    )

    return results_df


########## DICTIONARIES ###################
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

#### LOAD IN DATA #####

# Load in the iCGM Data
data = []
icgm_folder_name = "icgm-sensitivity-analysis-results-2020-11-02-nogit"
results_files_path = os.path.join("..", "..", "data", "processed", icgm_folder_name)

# Set where to save figures
save_fig_folder_name = icgm_folder_name

results_save_fig_path = os.path.join(
    "..",
    "..",
    "reports",
    "figures",
    "icgm-sensitivity-paired-comparison-figures",
    save_fig_folder_name,
)

not_pairwise_save_fig_path = os.path.join(
    "..",
    "..",
    "reports",
    "figures",
    "icgm-sensitivity-analysis-results-figures",
    save_fig_folder_name,
)

if not os.path.exists(results_save_fig_path):
    print("making directory " + results_save_fig_path + "...")
    os.makedirs(results_save_fig_path)

# icgm_results_df = create_data_frame_for_figures(
#     results_path=results_files_path,
#     save_path=results_save_fig_path,
#     results_folder_name=icgm_folder_name,
# )


# Load in the Ideal Sensor Data
data = []
ideal_sensor_folder_name = "icgm-sensitivity-analysis-results-2020-11-05-nogit"
baseline_files_path = os.path.join(
    "..", "..", "data", "processed", ideal_sensor_folder_name
)

# baseline_sensor_df = create_data_frame_for_figures(
#     is_baseline = True,
#     results_path=baseline_files_path,
#     save_path=results_save_fig_path,
#     results_folder_name=ideal_sensor_folder_name,
# )

#### CREATE FIGURES #####

#### Create pairwise figures ####
#run_pairwise_comparison(results_df=icgm_results_df, baseline_df=baseline_sensor_df, save_fig_folder_name=save_fig_folder_name)

#If have already loaded the data and just want pairwise comparison figures
#run_pairwise_comparison_figures(save_fig_folder_name=save_fig_folder_name)

# Create metadata tables
metadata_save_path = os.path.join(results_save_fig_path, "metadata_tables")
if not os.path.exists(metadata_save_path):
    print("making directory " + metadata_save_path + "...")
    os.makedirs(metadata_save_path)

icgm_results_df = pd.read_csv(os.path.join(results_save_fig_path, icgm_folder_name + "_results_df.csv"))

get_metadata_tables(icgm_results_df, fig_path=metadata_save_path)
