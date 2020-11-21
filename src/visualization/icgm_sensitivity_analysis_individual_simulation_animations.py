# %% REQUIRED LIBRARIES
import os
import pandas as pd

import tarfile

# from icgm_sensitivity_analysis_figures_and_tables import blood_glucose_risk_index, lbgi_risk_score, dka_index, dka_risk_score
from src.visualization.simulation_figures_shared_functions import data_loading_and_preparation
from src.visualization.simulation_figure_plotly import create_simulation_figure_plotly


bg_test_condition_to_check = "9"
LBGI_RS_to_check = 4
DKAI_RS_to_check = 0

# Uncomment if trying to search for which filenames to show
"""
animation_filenames = [] 

def get_data(filename, simulation_df):
    bg_test_condition = filename.split(".")[1].replace("bg", "")
    analysis_type = filename.split(".")[3]
    LBGI = blood_glucose_risk_index(bg_array=simulation_df["bg"])[0]
    LBGI_RS = lbgi_risk_score(LBGI)
    DKAI = dka_index(simulation_df["iob"], simulation_df["sbr"].iloc[0])
    DKAI_RS = dka_risk_score(DKAI)

    if (bg_test_condition == bg_test_condition_to_check) & (LBGI_RS == LBGI_RS_to_check) & (DKAI_RS == DKAI_RS_to_check):
        animation_filenames.append(filename)

    return [bg_test_condition, analysis_type, LBGI, LBGI_RS, DKAI, DKAI_RS]


data = []

path = os.path.join("..", "..", "data", "raw")
folder_name = "icgm-sensitivity-analysis-results-2020-09-19.tar.gz"
compressed_filestream = tarfile.open(os.path.join(path, folder_name))
file_list = [
    filename for filename in compressed_filestream.getnames() if ".csv" in filename
]

for i, filename in enumerate(file_list[0:10000]): #Change this when finish the figures
    simulation_df = pd.read_csv(compressed_filestream.extractfile(filename))
    # Add in the data
    print(i)
    data.append(get_data(filename, simulation_df))


columns = [
    "bg_test_condition",
    "analysis_type",
    "LBGI",
    "LBGI Risk Score",
    "DKAI",
    "DKAI Risk Score",
]

results_df = pd.DataFrame(data, columns=columns)

# rename the analysis types
results_df.replace({"tempBasal": "Temp Basal Analysis"}, inplace=True)
results_df.replace({"correctionBolus": "Correction Bolus Analysis"}, inplace=True)

print(animation_filenames)
"""

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
