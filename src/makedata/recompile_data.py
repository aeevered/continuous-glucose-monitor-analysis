import glob
import os
import json
import pandas as pd
import tidepool_data_science_metrics as metrics
from tidepool_data_science_models.models.icgm_sensor_generator_functions import (
    calc_mard,
    preprocess_data,
    calc_mbe,
    calc_icgm_sc_table,
    calc_icgm_special_controls_loss,
)
import numpy as np


def get_risk_metrics(true_bg_array, iob_array, sbr):
    lbgi, hbgi, bgri = metrics.glucose.blood_glucose_risk_index(true_bg_array)
    lbgi_rs = metrics.glucose.lbgi_risk_score(lbgi)
    dkai = metrics.insulin.dka_index(iob_array, sbr)
    dkai_rs = metrics.insulin.dka_risk_score(dkai)

    return lbgi, lbgi_rs, dkai, dkai_rs


def get_other_metrics(true_bg_array, icgm_array):

    prep_df = preprocess_data(
        true_bg_array, icgm_array.reshape(1, len(icgm_array)), icgm_range=[40, 400], ysi_range=[0, 900]
    )

    mard = calc_mard(prep_df)
    mbe = calc_mbe(prep_df)
    icgm_sc_table = calc_icgm_sc_table(prep_df)
    _, ind_percent_pass_test = calc_icgm_special_controls_loss(icgm_sc_table.dropna().round(), np.nan)

    icgm_lt_70 = icgm_array < 70
    abs_diff_lt_70 = np.abs(true_bg_array[icgm_lt_70] - icgm_array[icgm_lt_70])
    n_spurious_lt_70 = np.sum(abs_diff_lt_70 > 40)

    icgm_ge_70 = icgm_array >= 70
    percent_diff_ge_70 = icgm_array[icgm_ge_70] / true_bg_array[icgm_ge_70]
    n_spurious_ge_70 = np.sum((percent_diff_ge_70 < 0.6) | (percent_diff_ge_70 > 1.4))
    n_spurious = n_spurious_lt_70 + n_spurious_ge_70

    return mard, mbe, ind_percent_pass_test, n_spurious


sim_cols = [
    "bg",
    "bg_sensor",
    "iob",
    "temp_basal",
    "temp_basal_time_remaining",
    "sbr",
    "cir",
    "isf",
    "pump_sbr",
    "pump_isf",
    "pump_cir",
    "true_bolus",
    "true_carb_value",
    "true_carb_duration",
    "reported_bolus",
    "reported_carb_value",
    "reported_carb_duration",
    "delivered_basal_insulin",
    "undelivered_basal_insulin",
    "randint",
]

reg_path = "/Users/ed/projects/data-science--explore--risk-sim-figures/data/processed/icgm-sensitivity-analysis-results-2020-11-02-nogit"
spur_path = "/Users/ed/projects/data-science--explore--risk-sim-figures/data/processed/icgm-sensitivity-analysis-results-2020-11-04-spurious-nogit"

analysis_path = spur_path
# sim_tsv = sorted(glob.glob(os.path.join(analysis_path, "*.tsv")))

# %%
all_df = pd.DataFrame()
base_path = "/Users/ed/projects/data-science--explore--risk-sim-figures/data/processed/icgm-sensitivity-analysis-results-2020-11-05-baseline-nogit"
base_sim_tsv = sorted(glob.glob(os.path.join(base_path, "*.tsv")))
total_baselines = len(base_sim_tsv)

for s in range(total_baselines):
    print("starting {} of {}".format(s, total_baselines))
    s_tsv = base_sim_tsv[s]
    s_json = s_tsv.replace(".tsv", ".json")
    f = open(s_json, "r")
    json_data = json.loads(f.read())
    base_sim_id = json_data["sim_id"]
    vp, bg_condition, _, analysis_type = base_sim_id.split(".")

    base_df = pd.read_csv(s_tsv, sep="\t", low_memory=False, usecols=sim_cols)

    base_true_bg_array = base_df["bg"].values
    base_iob_array = base_df["iob"].values
    sbr = base_df["pump_sbr"].median()

    base_lbgi, base_lbgi_rs, base_dkai, base_dkai_rs = get_risk_metrics(base_true_bg_array, base_iob_array, sbr)

    for sensor_number in range(30):
        sim_id = "{}.{}.s{}.{}".format(vp, bg_condition, sensor_number, analysis_type)

        s_tsv = os.path.join(analysis_path, "{}.tsv".format(sim_id))
        s_json = s_tsv.replace(".tsv", ".json")

        f = open(s_json, "r")
        json_data = json.loads(f.read())

        assert sim_id == json_data["sim_id"]

        sensor_df = pd.DataFrame(json_data["patient"]["sensor"], index=[sim_id])
        controller_df = pd.DataFrame(json_data["controller"]).T
        controller_df.rename(index={"config": sim_id}, inplace=True)

        for key in json_data["patient"]["pump"]["config"].keys():
            val = float(json_data["patient"]["pump"]["config"][key]["schedule"][0]["setting"].strip("m").strip("g").strip("U"))
            controller_df[key] = val

        patient_df = pd.concat([controller_df, sensor_df], axis=1)

        patient_df["vp"] = vp
        patient_df["bg_condition"] = bg_condition
        patient_df["sensor_number"] = sensor_number
        patient_df["analysis_type"] = analysis_type

        # add baseline data
        patient_df["base_lbgi"] = base_lbgi
        patient_df["base_lbgi_rs"] = base_lbgi_rs
        patient_df["base_dkai"] = base_dkai
        patient_df["base_dkai_rs"] = base_dkai_rs

        assert sbr == patient_df.loc[sim_id, "basal_schedule"]

        sim_df = pd.read_csv(s_tsv, sep="\t", low_memory=False, usecols=sim_cols)
        true_bg_array = sim_df["bg"].values
        icgm_array = sim_df["bg_sensor"].values
        iob_array = sim_df["iob"].values

        lbgi, lbgi_rs, dkai, dkai_rs = get_risk_metrics(true_bg_array, iob_array, sbr)
        mard, mbe, ind_percent_pass_test, n_spurious = get_other_metrics(true_bg_array, icgm_array)

        patient_df["lbgi"] = lbgi
        patient_df["lbgi_rs"] = lbgi_rs
        patient_df["dkai"] = dkai
        patient_df["dkai_rs"] = dkai_rs

        patient_df["mard"] = mard
        patient_df["mbe"] = mbe
        patient_df["ind_percent_pass_test"] = ind_percent_pass_test
        patient_df["n_spurious"] = n_spurious

        patient_df["diff_lbgi_and_baseline"] = lbgi - base_lbgi

        all_df = pd.concat([all_df, patient_df])

all_df.to_csv("all_results_2020_11_08_spurious.csv")





# %% NOTES for Anne
# I was not able to get the same LBGI differences so I thought it would be good for me to independently recompile the data.
# [] a test can be written to make sure that all of the simulation parameters are identical
# * it looks like the max basal was set to 3x the scheduled basal rate
# [] add in mard and mbe to the metric toolbox
# [] also consider adding in a check for null values
# [] and consider a function that returns lbgi and lbgi_rs, same for dkai


# def generate_spurious_bg(true_bg_value, icgm_range=(40, 400)):

# %% check to see how many spurious events are in the dataset
# While it is possible for spurious events to result in values < 40 and > 400 mg/dL,
# these extreme values are outside of the device measurement range and therefore are not
# used in the calculation of meeting the iCGM specifications and will not be considered in this version.

