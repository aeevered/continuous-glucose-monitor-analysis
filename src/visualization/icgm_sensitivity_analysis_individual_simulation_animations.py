__author__ = "Anne Evered"


# %% REQUIRED LIBRARIES
import os
import pandas as pd
import numpy as np
from src.visualization.simulation_figures_shared_functions import data_loading_and_preparation
from src.visualization.simulation_figure_plotly import create_simulation_figure_plotly


# The purpose of this script is to generate animation or static plotly figures for simulation
# results from the iCGM Sensitivity Analysis.
# This script has been used in QA of iCGM Sensitivity Analysis results, for example to look at the
# actual simulation result traces for cases that have an elevated DKAI or LBGI risk score.
#


animation_filenames = [
    "vp57.bg4.sIdealSensor.correction_bolus.csv",
    "vp57.bg4.sIdealSensor.temp_basal_only.csv",
    "vp13.bg1.sIdealSensor.temp_basal_only.csv",
    "vp13.bg1.sIdealSensor.correction_bolus.csv",
    "vp85.bg7.sIdealSensor.meal_bolus.csv",
    "vp48.bg4.sIdealSensor.meal_bolus.csv",
]


# Show animation for the files
for filename in animation_filenames:  # [0:10]:

    path = os.path.join(
        "..",
        "..",
        "data",
        "raw",
        "icgm-sensitivity-analysis-results-2020-10-01-nogit",
    )

    simulation_df = data_loading_and_preparation(os.path.join(path, filename))

    traces = [{0: ["bg", "bg_sensor"], 1: ["sbr", "temp_basal_sbr_if_nan"], 2: ["iob"]}]

    print(simulation_df.columns)

    print(simulation_df["iob"])

    create_simulation_figure_plotly(
        files_need_loaded=False,
        data_frames=[simulation_df],
        file_location=path,
        file_names=[filename],
        traces=traces,
        subplots=3,
        time_range=(0, 8),
        main_title="<b>Example iCGM Simulation </b>",
        subtitle=filename,
        subplot_titles=[
            "BG Values",
            "Scheduled Basal Rate and Loop Decisions",
            "Insulin-on-Board",
        ],
        save_fig_path=os.path.join(
            "..",
            "..",
            "reports",
            "figures",
            "icgm-sensitivity-analysis-outlier-examples",
            "icgm-sensitivity-analysis-results-2020-10-01",
        ),
        figure_name="animation_" + filename,
        analysis_name="icmg_analysis",
        animate=True,
    )

########### Run some visualizations of specific scenario examples ###########
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



def create_sim_examples():

    fig_path = os.path.join(
        "..",
        "..",
        "reports",
        "figures",
        "icgm-sensitivity-paired-comparison-figures",
        save_fig_folder_name,
        "risk_score_change_example_figures"
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

    #1. Examples of high MARD (MARD >20)
    animation_filenames = combined_df.loc[
            (combined_df["mard_icgm"] > 40), "filename_icgm"
        ].tolist()[0:10]


    print("High MARD files:" + str(animation_filenames))
    print(len(animation_filenames))

    visualize_individual_sim_result(df = combined_df, icgm_path = results_files_path,
                                    baseline_path = baseline_files_path, save_fig_path = results_save_fig_path, save_fig_folder_name = "mard>40", animation_filenames = animation_filenames)


    #2. Examples of risk score bin changes > 2 buckets
    animation_filenames = combined_df.loc[
            (combined_df["LBGI Risk Score_icgm"] > combined_df["LBGI Risk Score_baseline"].apply(lambda x: x+1)), "filename_icgm"
        ].tolist()[0:10]

    print("LBGI jumps 2 risk bins files:" + str(animation_filenames))
    print(len(animation_filenames))

    visualize_individual_sim_result(df = combined_df, icgm_path = results_files_path,
                                    baseline_path = baseline_files_path, save_fig_path = results_save_fig_path, save_fig_folder_name = "LBGI_2_risk_bin_jumps", animation_filenames = animation_filenames)



    #3. Examples of really high LBGI

    animation_filenames = combined_df.loc[
            (combined_df["LBGI Difference"] > 8), "filename_icgm"
        ].tolist()[0:10]

    print("High LBGI files:" + str(animation_filenames))
    print(len(animation_filenames))

    visualize_individual_sim_result(df = combined_df, icgm_path = results_files_path,
                                    baseline_path = baseline_files_path, save_fig_path = results_save_fig_path
                                    , save_fig_folder_name = "LBGI_difference>8", animation_filenames = animation_filenames)


    #4. low bias high risk

    animation_filenames = combined_df.loc[
            ((combined_df["LBGI Difference"] > 6) & (combined_df["initial_bias_icgm"] < 5)), "filename_icgm"
        ].tolist()[0:10]


    visualize_individual_sim_result(df = combined_df, icgm_path = results_files_path,
                                    baseline_path = baseline_files_path, save_fig_path = results_save_fig_path
                                    , save_fig_folder_name = "low_bias_high_risk", animation_filenames = animation_filenames)



    #5. High MARD low risk

    animation_filenames = combined_df.loc[
        ((combined_df["LBGI Difference"] < 1) & (combined_df["mard_icgm"] > 30)), "filename_icgm"
        ].tolist()[0:10]

    visualize_individual_sim_result(df = combined_df, icgm_path = results_files_path,
                                    baseline_path = baseline_files_path, save_fig_path = results_save_fig_path
                                    , save_fig_folder_name = "high_mard_low_risk", animation_filenames = animation_filenames)




    #6. Any individual file want to look at
    # animation_filenames = []
    # visualize_individual_sim_result(df = combined_df, icgm_path = results_files_path,
    #                                 baseline_path = baseline_files_path, save_fig_path = results_save_fig_path
    #                                 , save_fig_folder_name = "other_misc_scenarios", animation_filenames = animation_filenames)


    return
