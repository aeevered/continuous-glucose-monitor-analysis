__author__ = "Anne Evered"

# This file contains functions that are used in both the plotly and matplotlib
# versions of simulation figure/animation.
# plotly version: simulation_figure_plotly.py
# matplotlib version: simulation_figure_matplotlib.py

# %% REQUIRED LIBRARIES
import pandas as pd
import numpy as np


def data_loading_and_preparation(filepath):
    """
    Load data and put in format needed for animation and figure of simulation output

    Parameters
    ----------
    filepath: str
        Path to the simulation output want to load data for

    Returns
    -------
    cleaned_sim_df: dataframe
        Cleaned dataframe of simulation results ready for use in the visualization functions.

    """

    # Read data based on file extension
    if filepath.endswith(".tsv"):
        raw_data_sim_df = pd.read_csv(filepath, sep="\t")
    else:
        raw_data_sim_df = pd.read_csv(filepath)

    # Create cleaned dataframe for animation of simulation output
    cleaned_sim_df = data_preparation(raw_data_sim_df)

    return cleaned_sim_df


def data_preparation(sim_df):
    """

    Parameters
    ----------
    sim_df: dataframe
        starting dataframe of simulation results

    Returns
    -------
    sim_df: dataframe
        cleaned dataframe of simulation results

    """

    # 1. Use index of dataframe to create minutes_post_simulation (each row is five minute)
    # 2. Create hours post simulation column for use in animation
    sim_df["five_minute_marks"] = sim_df.index
    sim_df["minutes_post_simulation"] = sim_df["five_minute_marks"].apply(
        lambda x: x * 5
    )
    sim_df["hours_post_simulation"] = sim_df["minutes_post_simulation"] / 60

    # Create column (temp_basal_sbr_if_nan) that uses the temp basal if there is a value
    # in that column and the scheduled basal rate if not; this column reflects what
    # basal rate is given in that moment (scheduled basal rate unless there is a temp basal)
    if "temp_basal" in sim_df.columns:
        sim_df["temp_basal_sbr_if_nan"] = sim_df["temp_basal"].mask(
            pd.isnull, sim_df["sbr"]
        )

    # TODO: generalize the above section so do not need to include an exception for the replay loop files
    # This section accomodates being able to create figure/animation of replay loop files
    # It accomplishes the same as above, but for the cases of jaeb_temp_basal and pyloopkit_temp_basal
    # column names that are present in the
    if "jaeb_temp_basal" in sim_df.columns:
        sim_df["jaeb_temp_basal_sbr_if_nan"] = sim_df["jaeb_temp_basal"].mask(
            pd.isnull, sim_df["sbr"]
        )

    if "pyloopkit_temp_basal" in sim_df.columns:
        sim_df["pyloopkit_temp_basal_sbr_if_nan"] = sim_df["pyloopkit_temp_basal"].mask(
            pd.isnull, sim_df["sbr"]
        )

    # Fill 0 values of certain columns with nans so they do not show up in visualizations.
    # For example, a true bolus value of 0 should not be shown for sake of clarity.
    cols = [
        "true_bolus",
        "suggested_bolus",
        "reported_bolus",
        "true_carb_value",
        "reported_carb_value",
        "true_carb_duration",
        "reported_carb_duration",
        "undelivered_basal_insulin",
    ]

    for col in cols:
        if col in sim_df.columns:
            sim_df[col] = sim_df[col].replace({0: np.nan})

    # Return the cleaned simulation dataframe

    return sim_df


# TODO: have the below function pull from design team tools api or other source of truth for Tidepool color scheme
#  and design features

# TODO: it may also make sense to have a separate function for matplotlib features and for plotly features
#   as some features are named differently in the matplotlib functions vs plotly functions. This version just
#   specifies a value for both names within the same dictionary. There may also be a better way to structure this
#   specifying of design choices.

# This is where the specific features are pulled from
def get_features_dictionary(field):
    """
    For a given field (bolus, temp_basal, etc), returns a dictionary of design elements
    (color, line style, etc) to use for that particular field. The purpose of this
    function is to maintain design consistency across visualizations, to match the general
    color scheme of other Tidepool products, and to have a single place to update when
    there is a desired change in a particular fields design style.

    Parameters
    ----------
    field: str
        field to get the features dictionary for

    Returns
    -------
    features_dictionary: dict
        dictionary of visualization features to be used in simulation figure/animations for that
        particular field


    """

    if field == "bg":
        features_dictionary = dict(
            legend_label="rTBG Trace (True BG)",
            color="#B1BEFF",
            alpha=1.0,
            linestyle="solid",
            linewidth=3,
            marker="None",
            markersize=0,
            drawstyle="default",
            mode="lines",
            dash=None,
            fill=None,
            shape="linear",
        )

    elif field == "bg_sensor":
        features_dictionary = dict(
            legend_label=" iCGM",
            color="#6AA84F",
            alpha=1.0,
            linestyle="None",
            linewidth=0,
            marker="o",
            markersize=3,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "sbr":
        features_dictionary = dict(
            legend_label="Scheduled Basal Rate",
            color="black",
            alpha=1,
            linestyle="--",
            linewidth=2,
            marker="None",
            markersize=0,
            drawstyle="default",
            mode="lines",
            dash="dash",
            fill=None,
            shape="hv",
        )

    elif field == "iob":
        features_dictionary = dict(
            legend_label="Insulin On Board",
            color="#744AC2",
            alpha=1,
            linestyle="solid",
            linewidth=2,
            marker="None",
            markersize=0,
            drawstyle="default",
            mode="lines",
            dash=None,
            fill=None,
            shape="linear",
        )

    elif field == "temp_basal_sbr_if_nan":
        features_dictionary = dict(
            legend_label="Loop Decision",
            color="#008ECC",
            alpha=1,
            linestyle="solid",
            step="pre",
            drawstyle="steps-pre",
            linewidth=2,
            marker="o",
            markersize=2,
            mode="lines",
            dash=None,
            fill="tozeroy",
            shape="hv",
        )

    elif field == "pyloopkit_temp_basal_sbr_if_nan":
        features_dictionary = dict(
            legend_label="Loop Decision (PyLoopKit)",
            color="#008ECC",
            alpha=1,
            linestyle="solid",
            step="pre",
            drawstyle="steps-pre",
            linewidth=2,
            marker="o",
            markersize=2,
            mode="lines",
            dash=None,
            fill="tozeroy",
            shape="hv",
        )

    elif field == "jaeb_temp_basal_sbr_if_nan":
        features_dictionary = dict(
            legend_label="Loop Decision (Jaeb Data)",
            color="purple",
            alpha=1,
            linestyle="solid",
            step="pre",
            drawstyle="steps-pre",
            linewidth=2,
            marker="o",
            markersize=2,
            mode="lines",
            dash=None,
            fill="tozeroy",
            shape="hv",
        )

    elif field == "delivered_basal_insulin":
        features_dictionary = dict(
            legend_label="Delivered Basal Insulin",
            color="#f9706b",
            alpha=0.4,
            linestyle="solid",
            linewidth=1,
            marker="o",
            markersize=2,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    # TODO: update with specific feature attributes want to use for fields below if going to include them in any
    #  animations or figures

    elif field == "undelivered_basal_insulin":
        features_dictionary = dict(
            legend_label="Undelivered Basal Insulin",
            color="#DB2E2C",
            alpha=1,
            linestyle="--",
            linewidth=2,
            marker="None",
            markersize=0,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "pump_sbr":
        features_dictionary = dict(
            legend_label="Pump Scheduled Basal Rate",
            color="#0068B5",
            alpha=1,
            linestyle="--",
            linewidth=2,
            marker="None",
            markersize=0,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "cir":
        features_dictionary = dict(
            legend_label="Carb-Insulin-Ratio",
            color="#f9706b",
            alpha=1.0,
            linestyle="None",
            linewidth=0,
            marker="o",
            markersize=8,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "pump_cir":
        features_dictionary = dict(
            legend_label="Pump Carb-Insulin-ratio",
            color="#40EBF9",
            alpha=1.0,
            linestyle="None",
            linewidth=0,
            marker="o",
            markersize=8,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "isf":
        features_dictionary = dict(
            legend_label="Insulin-Sensitivity-Factor",
            color="#f9706b",
            alpha=1.0,
            linestyle="None",
            linewidth=0,
            marker="o",
            markersize=8,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "pump_isf":
        features_dictionary = dict(
            legend_label="Pump Insulin-Sensitivity-Factor",
            color="#40EBF9",
            alpha=1.0,
            linestyle="None",
            linewidth=0,
            marker="o",
            markersize=8,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "reported_bolus":
        features_dictionary = dict(
            legend_label="Reported Bolus Value",
            color="#0068B5",
            alpha=1.0,
            linestyle="None",
            linewidth=0,
            marker="o",
            markersize=8,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "true_bolus":
        features_dictionary = dict(
            legend_label="True Bolus Value",
            color="#4ce791",
            alpha=1.0,
            linestyle="None",
            linewidth=0,
            marker="o",
            markersize=8,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "suggested_bolus":
        features_dictionary = dict(
            legend_label="Suggested Bolus Value",
            color="#4ce791",
            alpha=1.0,
            linestyle="None",
            linewidth=0,
            marker="o",
            markersize=8,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "true_carb_value":
        features_dictionary = dict(
            legend_label="True Carb Value",
            color="#f95f3b",
            alpha=1.0,
            linestyle="None",
            linewidth=0,
            marker="o",
            markersize=8,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "reported_carb_value":
        features_dictionary = dict(
            legend_label="Reported Carb Value",
            color="#4ce791",
            alpha=1.0,
            linestyle="None",
            linewidth=0,
            marker="o",
            markersize=8,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "true_carb_duration":
        features_dictionary = dict(
            legend_label="True Carb Duration",
            color="#f9706b",
            alpha=1.0,
            linestyle="None",
            linewidth=0,
            marker="o",
            markersize=8,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    elif field == "reported_carb_duration":
        features_dictionary = dict(
            legend_label="Reported Carb Duration",
            color="#40EBF9",
            alpha=1.0,
            linestyle="None",
            linewidth=0,
            marker="0",
            markersize=8,
            drawstyle="default",
            mode="markers",
            dash="dash",
            fill=None,
            shape="linear",
        )

    return features_dictionary
