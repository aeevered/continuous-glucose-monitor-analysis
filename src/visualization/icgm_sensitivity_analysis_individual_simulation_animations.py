__author__ = "Anne Evered"


# %% REQUIRED LIBRARIES
import os
import pandas as pd
import numpy as np
from src.visualization.simulation_figures_shared_functions import (
    data_loading_and_preparation,
)
from src.visualization.simulation_figure_plotly import create_simulation_figure_plotly


# The purpose of this script is to generate animation or static plotly figures for simulation
# results from the iCGM Sensitivity Analysis. This script has been used in QA of iCGM Sensitivity
# Analysis results, for example to look at the actual simulation result traces for cases that have
# an elevated DKAI or LBGI risk score. It could also be used moving forward to generate example
# simulation figures for inclusion in analysis report.


# There are two options provided here: you can either add a list of the particular filenames you
# want to visualize, or there is a function create_visualizations_of_sims_with_particular_criteria
# that will create visualizations based on criteria we have been particularly interested in looking
# at (high mard, risk score bin changes, etc.). Additional criteria can be added to this function, as
# needed.

# These functions depend on the aggregate pairwise_comparison_combined_df_... being loaded for the
# particular results/baseline files first. See: icgm_sensitivity_analysis_report_figures_and_tables.py


def visualize_individual_icgm_analysis_simulations(
    df,
    icgm_path,
    baseline_path,
    save_fig_path,
    save_folder_name,
    filenames=[],
):
    """

    Generate animation or static plotly figures for simulation results from the
    iCGM Sensitivity Analysis based on passed in file names and file locations.

    Parameters
    ----------
    df: dataframe
        Combined (pairwise) results dataframe. This is used for accessing
        certain metrics for the particular dataframe.
    icgm_path: str
        File path where the individual icgm simulation results files are located.
    baseline_path: str
        File path where the individual baseline simulation results files are located.
    save_fig_path: str
        File path of where to save the figures to.
    save_folder_name: str
        Name of the folder within the file path to save the figures
        File path of where to save the figures to.

    Returns
    -------

    """

    # Only visualize the first 10 filenames for the sake of speed
    for i, filename in enumerate(filenames[0:10]):

        print(i, filename)

        # Get various fields from aggregated results files to put in figure title
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

        mard_icgm = df.loc[df["filename_icgm"] == filename, "mard_icgm"].iloc[0]

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

        # Get max and min values to use for custom axis ranges
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

        # Create and save simulation figure
        create_simulation_figure_plotly(
            files_need_loaded=False,
            data_frames=[icgm_simulation_df, baseline_simulation_df],
            file_location=os.path.join("..", "..", "data", "raw"),
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
                save_folder_name,
            ),
            figure_name=filename,
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


def create_visualizations_of_sims_with_particular_criteria(
    combined_df, results_save_fig_path, baseline_files_path, results_files_path
):
    """

    Create visualizations based on simulation criteria.
    The purpose of this function is to be able to look into specific simulations in categories
    we are particularly interested in either for QA or results interpretation/final report).

    Currently, these particular categories of cases are:
        1. Examples of high MARD (MARD >20)
        2. Examples of risk score bin changes > 2 buckets
        3. Examples of really high LBGI
        4. Examples of low bias high risk
        5. Examples of high MARD low risk

    Parameters
    ----------
    combined_df: dataframe
        Combined (pairwise) results dataframe. This is used for accessing
        certain metrics for the particular dataframe.
    results_save_fig_path: str
        File path of where to save the figures to.
    baseline_files_path: str
        File path where the individual baseline simulation results files are located.
    results_files_path: str
        File path where the individual icgm simulation results files are located.

    Returns
    -------

    """

    # 1. Examples of high MARD (MARD >20)
    animation_filenames = combined_df.loc[
        (combined_df["mard_icgm"] > 40), "filename_icgm"
    ].tolist()[0:10]

    print("High MARD files:" + str(animation_filenames))
    print(len(animation_filenames))

    visualize_individual_icgm_analysis_simulations(
        df=combined_df,
        icgm_path=results_files_path,
        baseline_path=baseline_files_path,
        save_fig_path=results_save_fig_path,
        save_folder_name="mard>40",
        filenames=animation_filenames,
    )

    # 2. Examples of risk score bin changes > 2 buckets
    animation_filenames = combined_df.loc[
        (
            combined_df["LBGI Risk Score_icgm"]
            > combined_df["LBGI Risk Score_baseline"].apply(lambda x: x + 1)
        ),
        "filename_icgm",
    ].tolist()[0:10]

    print("LBGI jumps 2 risk bins files:" + str(animation_filenames))
    print(len(animation_filenames))

    visualize_individual_icgm_analysis_simulations(
        df=combined_df,
        icgm_path=results_files_path,
        baseline_path=baseline_files_path,
        save_fig_path=results_save_fig_path,
        save_folder_name="LBGI_2_risk_bin_jumps",
        filenames=animation_filenames,
    )

    # 3. Examples of really high LBGI

    animation_filenames = combined_df.loc[
        (combined_df["LBGI Difference"] > 8), "filename_icgm"
    ].tolist()[0:10]

    print("High LBGI files:" + str(animation_filenames))
    print(len(animation_filenames))

    visualize_individual_icgm_analysis_simulations(
        df=combined_df,
        icgm_path=results_files_path,
        baseline_path=baseline_files_path,
        save_fig_path=results_save_fig_path,
        save_folder_name="LBGI_difference>8",
        filenames=animation_filenames,
    )

    # 4. low bias high risk

    animation_filenames = combined_df.loc[
        ((combined_df["LBGI Difference"] > 6) & (combined_df["initial_bias_icgm"] < 5)),
        "filename_icgm",
    ].tolist()[0:10]

    visualize_individual_icgm_analysis_simulations(
        df=combined_df,
        icgm_path=results_files_path,
        baseline_path=baseline_files_path,
        save_fig_path=results_save_fig_path,
        save_folder_name="low_bias_high_risk",
        filenames=animation_filenames,
    )

    # 5. High MARD low risk

    animation_filenames = combined_df.loc[
        ((combined_df["LBGI Difference"] < 1) & (combined_df["mard_icgm"] > 30)),
        "filename_icgm",
    ].tolist()[0:10]

    visualize_individual_icgm_analysis_simulations(
        df=combined_df,
        icgm_path=results_files_path,
        baseline_path=baseline_files_path,
        save_fig_path=results_save_fig_path,
        save_folder_name="high_mard_low_risk",
        filenames=animation_filenames,
    )

    return


if __name__ == "__main__":

    # Specify the iCGM data filepath
    icgm_folder_name = "icgm-sensitivity-analysis-results-2020-11-02-nogit"
    results_files_path = os.path.join("..", "..", "data", "raw", icgm_folder_name)

    # Specify the Baseline data fildepath
    ideal_sensor_folder_name = "icgm-sensitivity-analysis-results-2020-11-05-nogit"
    baseline_files_path = os.path.join(
        "..", "..", "data", "raw", ideal_sensor_folder_name
    )

    # Set where to save figures
    save_fig_folder_name = icgm_folder_name

    # Specify where to save files
    results_save_fig_path = os.path.join(
        "..",
        "..",
        "reports",
        "figures",
        "icgm-sensitivity-paired-comparison-figures",
        save_fig_folder_name,
    )

    # Load in combined df
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

    # Visualize specific filenames

    # Update animation_filenames with whatever files you want to visualize
    filenames = [
        "vp24b7a7a140092b0d2b1c4754efdd06a832493da1a9af866b304d16113de7abeb.bg9.s20.meal_bolus.tsv"
    ]
    visualize_individual_icgm_analysis_simulations(
        df=combined_df,
        icgm_path=results_files_path,
        baseline_path=baseline_files_path,
        save_fig_path=results_save_fig_path,
        save_folder_name="other_misc_scenarios",
        filenames=filenames,
    )

    # Create visualizations based on particular simulation criteria
    create_visualizations_of_sims_with_particular_criteria(
        combined_df, results_save_fig_path, baseline_files_path, results_files_path
    )
