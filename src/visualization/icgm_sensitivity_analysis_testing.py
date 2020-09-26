# %% REQUIRED LIBRARIES
import os
import pandas as pd
import json
from save_view_fig import save_view_fig

#### Look for suspect results files ####

temp_basal_files_with_true_bolus = []
meal_bolus_files_with_no_true_bolus = []
correction_bolus_files_with_no_true_bolus = []
bg_greater_thousand = []

results_path = os.path.join("..", "..", "data", "processed", "icgm-sensitivity-analysis-results-2020-09-19-nogit")


for i, filename in enumerate(os.listdir(results_path)[0:0]):
    if filename.endswith(".csv"):
        print(i, filename)
        simulation_df = pd.read_csv(os.path.join(results_path, filename))

        if 'temp' in filename:
            if not (simulation_df['true_bolus'] == 0).all():
                temp_basal_files_with_true_bolus.append(filename)

        if 'correction' in filename:
            if (simulation_df['true_bolus'] == 0).all():
                correction_bolus_files_with_no_true_bolus.append(filename)

        if 'meal' in filename:
            if (simulation_df['true_bolus'] == 0).all():
                meal_bolus_files_with_no_true_bolus.append(filename)

        if (simulation_df['bg'] > 1000).any():
            bg_greater_thousand.append(filename)


df1 = pd.DataFrame.from_dict({'temp_basal_files_with_true_bolus': temp_basal_files_with_true_bolus})
df2 = pd.DataFrame.from_dict({'meal_bolus_files_with_no_true_bolus': meal_bolus_files_with_no_true_bolus})
df3 = pd.DataFrame.from_dict({'correction_bolus_files_with_no_true_bolus': correction_bolus_files_with_no_true_bolus})
df4 = pd.DataFrame.from_dict({'files_where_bg_greater_than_one_thousand': bg_greater_thousand})


files_df = pd.concat([df1, df2, df3, df4], ignore_index=True, axis=1)

files_df = files_df.rename(columns={0: "temp_basal_files_with_true_bolus",
                                    1: "meal_bolus_files_with_no_true_bolus",
                                    2: "correction_bolus_files_with_no_true_bolus",
                                    3: "files_where_bg_greater_than_one_thousand"})
print(files_df)

#files_df.to_csv(".../Tidepool 2020/Tidepool Repositories/data-science--explore--risk-sim-figures/data/processed/suspect_icgm_analysis_files_all.csv")


#### Look for scenario files outside of the clinical recommendations ####

scenarios_path = os.path.join("..", "..", "data", "processed", "icgm-sensitivity-analysis-scenarios-2020-07-10-nogit")

scenarios_with_settings_outside_clinical_bounds = []
virtual_patients = []
isf = []
cir = []
br = []
max_bolus = []

def outside_clinical_bounds(df):
    if float(df.loc[df['setting_name'] == 'sensitivity_ratio_values', '0'].iloc[0]) < 10 or float(df.loc[df['setting_name'] == 'sensitivity_ratio_values', '0'].iloc[0]) > 500:
        return True
    elif float(df.loc[df['setting_name'] == 'carb_ratio_values', '0'].iloc[0]) < 2 or float(df.loc[df['setting_name'] == 'carb_ratio_values', '0'].iloc[0]) > 150:
        return True
    elif float(df.loc[df['setting_name'] == 'basal_rate_values', '0'].iloc[0]) < 0.05 or float(df.loc[df['setting_name'] == 'basal_rate_values', '0'].iloc[0]) > 30:
        return True
    elif float(df.loc[df['setting_name'] == 'max_bolus', 'settings'].iloc[0]) < 0 or float(df.loc[df['setting_name'] == 'max_bolus', 'settings'].iloc[0]) > 30:
        return True
    return False


for i, scenario_filename in enumerate(os.listdir(scenarios_path))[0:0]):
    if scenario_filename.endswith(".csv"):
        print(i, scenario_filename)
        scenario_df = pd.read_csv(os.path.join(scenarios_path, scenario_filename))
        if outside_clinical_bounds(scenario_df):
            isf.append(float(scenario_df.loc[scenario_df['setting_name'] == 'sensitivity_ratio_values', '0'].iloc[0]))
            cir.append(float(scenario_df.loc[scenario_df['setting_name'] == 'carb_ratio_values', '0'].iloc[0]))
            br.append(float(scenario_df.loc[scenario_df['setting_name'] == 'basal_rate_values', '0'].iloc[0]))
            max_bolus.append(float(scenario_df.loc[scenario_df['setting_name'] == 'max_bolus', 'settings'].iloc[0]))
            scenarios_with_settings_outside_clinical_bounds.append(scenario_filename)
            for i, results_filename in enumerate(os.listdir(results_path)):
                if (results_filename.endswith(".json")) and (len(results_filename) < 10):
                    f = open(os.path.join(results_path, results_filename), "r")
                    json_data = json.loads(f.read())
                    patient_characteristics_df = pd.DataFrame(json_data, index=['i', ])
                    if patient_characteristics_df["patient_scenario_filename"].iloc[0].split("/")[-1].split("_")[1] == scenario_filename.split("_")[1]:
                        virtual_patients.append(results_filename)


scenarios_outside_clinical_bounds_df = pd.DataFrame(list(zip(scenarios_with_settings_outside_clinical_bounds, virtual_patients, isf, cir, br, max_bolus)), columns =['scenarios_with_settings_outside_clinical_bounds', 'virtual_patients', "isf", "cir", "br", "max_bolus"])

#scenarios_outside_clinical_bounds_df.to_csv("/Users/anneevered/Desktop/Tidepool 2020/Tidepool Repositories/data-science--explore--risk-sim-figures/data/processed/scenarios_outside_clinical_bounds.csv")







