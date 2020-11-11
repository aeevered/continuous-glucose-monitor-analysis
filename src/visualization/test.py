import os
import json
import difflib
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
from plotly.subplots import make_subplots
from risk_scenario_figures_plotly import create_simulation_figure_plotly
from risk_scenario_figures_shared_functions import data_loading_and_preparation

#vpfefe484d76b59eb7706124bb34e82291c4e9857bb95ddceb440c0163083437ae

# this vpid is not linking to any of the sensor ids

#iterate through all of the virtual patient files and see if can find that id

file_path = os.path.join("..", "..", "data", "processed", "icgm-sensitivity-analysis-results-2020-09-19-nogit")

for file in sorted(os.listdir(file_path)):
    if file.endswith(".json") and len(file) < 12:
        print(file)
        f = open(
            os.path.join(
                file_path,
                file
            ),
            "r",
        )
        json_data = json.loads(f.read())
        #print(json_data)
        patient_id = json_data["patient_scenario_filename"].split("/")[-1].split(".")[0].replace("train_", "")
        print(patient_id)
        # if patient_id == "fefe484d76b59eb7706124bb34e82291c4e9857bb95ddceb440c0163083437ae":
        #     print("matches: " + file)


#Also merging on analysis type and on bg_test_condition: ["virtual_patient_num", "analysis_type", "bg_test_condition"]

#Conclusion is for that patient id, which is patient 99 - they only have rows in the baseline dataset for condition 1 and 2
#(though in the raw results files there are other conditions); need to see why those are not being captured

# results_path = os.path.join("..", "..", "data", "processed", "icgm-sensitivity-analysis-results-2020-11-02-nogit")
#
# for i, filename in enumerate(sorted(os.listdir(results_path))):
#     if filename.endswith(".tsv"):
#         simulation_df = pd.read_csv(
#             os.path.join(results_path, filename), sep="\t"
#         )
#         print(simulation_df.loc[0]["bg"])
#         print(simulation_df.loc[1]["bg"])
#         assert (simulation_df.loc[0]["bg"] == simulation_df.loc[1]["bg"]), "First two BG values of simulation are not equal"

df = pd.read_csv("/Users/anneevered/Desktop/Tidepool 2020/Tidepool Repositories/data-science--explore--risk-sim-figures/data/processed/icgm-sensitivity-analysis-results-2020-10-06-nogit/vp44.bg9.sIdealSensor.meal_bolus.csv")
LBGI = metrics.glucose.blood_glucose_risk_index(bg_array=df["bg"])[0]
print(df["bg"])
print(LBGI)

df = pd.read_csv("/Users/anneevered/Desktop/Tidepool 2020/Tidepool Repositories/data-science--explore--risk-sim-figures/data/processed/icgm-sensitivity-analysis-results-2020-11-05-nogit/vp80a5c60283c2b095d69cca4f64c26e2564958a07e2f0e19fafd073ed47d2b5e7.bg9.s0.meal_bolus.tsv", sep="\t")
LBGI = metrics.glucose.blood_glucose_risk_index(bg_array=df["bg"])[0]
print(df["bg"])
print(LBGI)