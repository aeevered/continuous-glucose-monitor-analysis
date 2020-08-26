# %% REQUIRED LIBRARIES
import os
import pandas as pd

import tarfile
from icgm_sensitivity_analysis_figures_and_tables import blood_glucose_risk_index, lbgi_risk_score, dka_index, dka_risk_score
from risk_scenario_figures_plotly import create_simulation_figure_plotly
from risk_scenario_figures_shared_functions import data_loading_and_preparation


bg_test_condition_to_check="2"
LBGI_RS_to_check=1
DKAI_RS_to_check=0
animation_filenames = ['icgm-sensitivity-analysis-results-2020-07-12/vp9.bg2.s15.temp_basal_only.csv']

#Uncomment if trying to search for which filenames to show
'''
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

path = os.path.join("..", "..", "data", "processed")
folder_name = "icgm-sensitivity-analysis-results-ALL-RESULTS-2020-07-12.gz"
compressed_filestream = tarfile.open(os.path.join(path, folder_name))
file_list = [
    filename for filename in compressed_filestream.getnames() if ".csv" in filename
]

for i, filename in enumerate(file_list[0:0]): #Change this when finish the figures
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

'''


#Show animation for the files
for filename in animation_filenames[0:10]:

    path = os.path.join("..", "..", "data", "processed")
    folder_name = "icgm-sensitivity-analysis-results-ALL-RESULTS-2020-07-12.gz"
    compressed_filestream = tarfile.open(os.path.join(path, folder_name))

    simulation_df = data_loading_and_preparation(compressed_filestream.extractfile(filename))

    traces = [{0: ["bg", "bg_sensor"], 1: ["sbr", "temp_basal_sbr_if_nan"]}]

    print(simulation_df.columns)

    create_simulation_figure_plotly(
        files_need_loaded=False,
        data_frames=[simulation_df],
        file_location=path,
        file_names=[filename],
        traces=traces,
        subplots=2,
        time_range=(0, 8),
        main_title="<b>Example iCGM Simulation </b>",
        subtitle="(LBGI Risk Score: " + str(LBGI_RS_to_check) + "; DKAI Risk Score: " + str(DKAI_RS_to_check) +")",
        subplot_titles=[
            "BG Values",
            "Scheduled Basal Rate and Loop Decisions",
        ],
        save_fig_path=os.path.join("..", "..", "reports", "figures","icgm_analysis_bg_test_condition_9"),
        figure_name="animation_"+filename,
        analysis_name="icmg_analysis",
        animate=True,
    )