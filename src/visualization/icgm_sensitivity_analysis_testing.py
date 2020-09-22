# %% REQUIRED LIBRARIES
import os
import pandas as pd


temp_basal_files_with_true_bolus = []
meal_bolus_files_with_no_true_bolus = []
correction_bolus_files_with_no_true_bolus = []
bg_greater_thousand = []

path = os.path.join("..", "..", "data", "processed", "icgm-sensitivity-analysis-results-2020-09-19-nogit")


for i, filename in enumerate(os.listdir(path)): #[0:10]):
    if filename.endswith(".csv"):
        print(i, filename)
        simulation_df = pd.read_csv(os.path.join(path, filename))

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

files_df.to_csv("/Users/anneevered/Desktop/Tidepool 2020/Tidepool Repositories/data-science--explore--risk-sim-figures/data/processed/suspect_icgm_analysis_files_all.csv")
