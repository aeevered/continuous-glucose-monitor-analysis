"""
this file calculates all of the icgm sensitivity analysis statistics (AKA metrics)
"""

# %% REQUIRED LIBRARIES
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio
import datetime as dt

utc_string = dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%m-%S")
# TODO: automatically grab the code version to add to the figures generated
code_version = "v0-1-0"


# %% FUNCTIONS
# TODO: us mypy and specify the types
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
            font_size=12,
            height=cell_height[0],
        ),
    )

    if len(cell_width) > 1:
        table_width = np.sum(np.asarray(cell_width))
    else:
        table_width = n_cols * cell_width[0]
    table_height = (n_rows + 1.5) * cell_height[0] + cell_header_height[0]
    table_layout = go.Layout(margin=dict(l=10, r=10, t=10, b=0), width=table_width, height=table_height)
    fig = go.Figure(data=_table, layout=table_layout)
    if view_fig:
        plot(fig)

    file_name = "{}-{}_{}_{}".format(analysis_name, table_name, utc_string, code_version)
    if save_fig:
        pio.write_image(
            fig=fig, file=os.path.join(save_fig_path, file_name + ".{}".format(image_type)), format=image_type
        )
    if save_csv:
        table_df.to_csv(os.path.join(save_fig_path, file_name + ".csv"))

    return


# adding in some generic methods for tables based on bins
def bin_data(bin_breakpoints):
    # the bin_breakpoints are the points that are greater than or equal to
    return pd.IntervalIndex.from_breaks(bin_breakpoints, closed="left")


# %% LOAD IN DATA
# Read in data one chunk at a time, getting only the columns we need
sim_results_location = os.path.join("..", "..", "data", "processed")
simulation_file = "risk-sim-results-2020-04-13"
file_import_path = os.path.abspath(os.path.join(sim_results_location, simulation_file))
demographic_datapath = os.path.join(file_import_path + "-just-demographics-nogit.csv")
sensor_datapath = os.path.join(file_import_path + "-sensor-metrics-nogit.csv")
summary_datapath = os.path.join(file_import_path + "-summary-metrics-nogit.csv")

# reduced_data_path_and_file_name = os.path.join(file_import_path + "-reduced.csv")
if os.path.exists(demographic_datapath):
    demographic_df = pd.read_csv(demographic_datapath, index_col=[0])
    sensor_df = pd.read_csv(sensor_datapath, index_col=[0])
    summary_df = pd.read_csv(sensor_datapath, index_col=[0])
else:
    file_reader = pd.read_csv(file_import_path + ".gz", low_memory=False, chunksize=10000)
    chunk_num = 1
    sim_data = []
    for chunk in file_reader:
        print("Appending chunk # " + str(chunk_num))
        sim_data.append(chunk)
        chunk_num += 1

    all_data_df = pd.concat(sim_data)

    # all data frames should include the following:
    essential_columns = [
        "analysis_type",
        "virtual_patient_num",
        "bg_test_condition",
        "icgm_sensor_num",
        "icgm_test_condition",
    ]

    # save smaller datasets for quicker processing
    demographic_columns = essential_columns + [
        "age",
        "ylw",
        "CIR",
        "ISF",
        "BR",
    ]
    demographic_df = all_data_df[demographic_columns]
    demographic_df.to_csv(demographic_datapath)

    # JUST SENSOR DATA
    # TODO: coordinate with jason about different approach to column names and/or storing data
    # FIXME: this method of getting the columns will break if/when the output data order changes
    sensor_columns = all_data_df.columns[1:165]
    sensor_df = all_data_df[sensor_columns]
    sensor_df.to_csv(sensor_datapath)

    # DATA FOR SUMMARY RESULTS TABLE
    # TODO: coordinate with jason about different approach to column names and/or storing data
    # FIXME: this method of getting the columns will break if/when the output data order changes
    summary_columns = all_data_df.columns[1:22].tolist() + ["ICGM_PASS%"]
    summary_df = all_data_df[summary_columns]
    summary_df.to_csv(summary_datapath)

    # save data_df for quick load
    # data_df.to_csv(reduced_data_path_and_file_name, index=False)

# %% get data for russ
make_data_sample = False
if make_data_sample:
    data_sample_all = pd.read_csv(file_import_path, low_memory=False, nrows=100,)
    bg_actual_sample = data_sample_all.loc[0:0, data_sample_all.columns.str.contains("bg_actual")]
    bg_actual_sample.T.to_csv(
        os.path.join("..", "..", "data", "processed", "bg-array-sample.csv"), header=False, index=False,
    )

    bg_actual_sample = data_sample_all.loc[:, data_sample_all.columns.str.contains("bg_actual")]
    bg_actual_sample.T.to_csv(
        os.path.join("..", "..", "data", "processed", "bg-matrix-sample-each-col-is-a-unique-time-series.csv",),
        header=False,
        index=False,
    )

# %% prepare demographic data for tables
virtual_patient_group = demographic_df.groupby("virtual_patient_num")
demographic_reduced_df = virtual_patient_group[["age", "ylw", "CIR", "ISF", "BR"]].median()
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
)
# %% Carb to Insulin Ratio Table
cir_bin_breakpoints = np.array(
    [demographic_reduced_df["CIR"].min(), 5, 10, 15, 20, 25, demographic_reduced_df["CIR"].max() + 1,]
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
)

# %% Basal Rate (BR) Table
br_bin_breakpoints = np.append(np.arange(0, 1.5, 0.25), np.arange(1.5, demographic_reduced_df["BR"].max() + 0.5, 0.5),)
br_bins = bin_data(br_bin_breakpoints)

# make an br table
br_table = pd.DataFrame(index=br_bins.astype("str"))
br_table.index.name = "Basal Rate"

# cut the data by bin
demographic_reduced_df["br_bin"] = np.nan
demographic_reduced_df["br_bin"] = pd.cut(demographic_reduced_df["BR"], br_bins)
br_table["Count"] = demographic_reduced_df.groupby("br_bin")["BR"].count().values

# add in missing data
br_table.loc["Missing", "Count"] = demographic_reduced_df["BR"].isnull().sum()

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
)


# %% Summary Table
def get_summary_stats(df, level_of_analysis_name):
    # rename some of the columns
    df.rename(
        {"loop_LBGI": "LBGI", "loop_LBGI_RS": "LBGI Risk Score", "DKAI_RS": "DKAI Risk Score"},
        axis="columns",
        inplace=True,
    )

    # create a summary table
    # NOTE: there is a known bug with plotly tables https://github.com/plotly/plotly.js/issues/3251
    outcome_table_cols = [
        "Median LBGI<br>" "     (IQR)",  # adding in spacing because of bug
        "Median LBGI Risk Score<br>" "             (IQR)",  # adding in spacing because of bug
        "Median DKAI<br>" "     (IQR)",  # adding in spacing because of bug
        "Median DKAI Risk Score<br>" "             (IQR)",  # adding in spacing because of bug
    ]
    outcome_names = ["LBGI", "LBGI Risk Score", "DKAI", "DKAI Risk Score"]
    count_name = " Number of<br>Simulations"
    summary_table_cols = [count_name] + outcome_table_cols
    summary_table = pd.DataFrame(columns=summary_table_cols)
    summary_table.index.name = "Level of Analysis"

    for outcome, outcome_table_col in zip(outcome_names, outcome_table_cols):
        summary_stats = pd.Series(df[outcome].describe())
        summary_table.loc[level_of_analysis_name, count_name] = summary_stats["count"]
        summary_table.loc[level_of_analysis_name, outcome_table_col] = "{} (IQR={}-{})".format(
            summary_stats["50%"].round(1), summary_stats["25%"].round(1), summary_stats["75%"].round(1)
        )
    return summary_table


# %% first remove any/all iCGM sensor batches that did not meet iCGM special controls
summary_df_reduced = summary_df[summary_df["ICGM_PASS%"] == 100]

# first do all analyses
all_analyses_summary_df = get_summary_stats(summary_df_reduced, "All Analyses Combined")

# break up by analysis type
# rename the analysis types
summary_df_reduced.replace({"tempBasal": "Temp Basal Analysis"}, inplace=True)
summary_df_reduced.replace({"correctionBolus": "Correction Bolus Analysis"}, inplace=True)
summary_df_reduced.replace({"mealBolus": "Meal Bolus Analysis"}, inplace=True)

for analysis_type in summary_df_reduced["analysis_type"].unique():
    temp_df = summary_df_reduced[summary_df_reduced["analysis_type"] == analysis_type]
    temp_summary = get_summary_stats(temp_df, analysis_type)
    all_analyses_summary_df = pd.concat([all_analyses_summary_df, temp_summary])

# break up by bg test condition
for bg_test_condition in summary_df_reduced["bg_test_condition"].unique():
    temp_df = summary_df_reduced[summary_df_reduced["bg_test_condition"] == bg_test_condition]
    temp_summary = get_summary_stats(temp_df, "BG Test Condition {}".format(bg_test_condition))
    all_analyses_summary_df = pd.concat([all_analyses_summary_df, temp_summary])

# make table
make_table(
    all_analyses_summary_df.reset_index(),
    table_name="summary-risk-table",
    analysis_name="icgm-sensitivity-analysis",
    cell_header_height=[60],
    cell_height=[30],
    cell_width=[200, 100, 150, 200, 150, 200],
    image_type="png",
    view_fig=True,
    save_fig=True,
)

# %% Visualization Functions
def make_boxplot(
    table_df,
    image_type="png",
    figure_name="<number-or-name>-boxplot",
    analysis_name="analysis-<name>",
    metric="LBGI",
    level_of_analysis= "analysis_type",
    notched_boxplot=True,
    y_scale_type="linear",
    view_fig=True,
    save_fig=True,
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
):
    """
    Create a boxplot figure.

    :param table_df: Table name of data to visualize.
    :param image_type: Image type for saving image (eg. png, jpeg).
    :param figure_name: Name of figure (for name of file for saving figure).
    :param analysis_name: Name of analysis (for name of file for saving figure).
    :param metric: Metric from table_df to visualize on the y-axis.
    :param level_of_analysis: Level of analysis breakdown ("all", "bg_test_condition", etc.) for x-axis.
    :param notched_boxplot: True if want the boxplot to be notched boxplot style.
    :param y_scale_type: Log or linear for y axis scale.
    :param view_fig: True if want to view figure.
    :param save_fig: True if want to save figure.
    :param save_fig_path: File path for where to save figure.
    :return:
    """

    #If level_of_analysis is to show all analyses (no breakdown), show as single box.
    if level_of_analysis == "all":
        fig = px.box(table_df,
                     x=None,
                     y=metric,
                     color_discrete_sequence=px.colors.qualitative.T10,
                     notched = notched_boxplot)

    #Otherwise show separate boxplot for each breakdown category.
    else:
        fig = px.box(table_df,
                     x=level_of_analysis,
                     y=metric,
                     color= level_of_analysis,
                     color_discrete_sequence=px.colors.qualitative.T10,
                        #can also explicitly define the sequence: ["red", "green", "blue"],
                     notched = notched_boxplot)

    #Update figure layout based on scale type
    fig.update_layout(yaxis_type=y_scale_type)

    save_view_fig(fig, image_type, figure_name, analysis_name, view_fig, save_fig, save_fig_path)

    return


def make_bubble_plot(
    table_df,
    image_type="png",
    figure_name="<number-or-name>-bubbleplot",
    analysis_name="analysis-<name>",
    metric="LBGI",
    level_of_analysis= "analysis_type",
    view_fig=True,
    save_fig=True,
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
):
    if level_of_analysis == "all":

        df = table_df[[metric]]
        grouped_df = df.groupby([metric]).size().reset_index(name="count")

        fig = px.scatter(
            x=None,
            y=grouped_df[metric],
            size=grouped_df["count"],
            color=grouped_df["count"],
            size_max=40)

    else:

        df = table_df[[level_of_analysis, metric]]
        grouped_df = df.groupby([level_of_analysis, metric]).size().reset_index(name="count")

        fig = px.scatter(
            x=grouped_df[level_of_analysis],
            y=grouped_df[metric],
            size=grouped_df["count"],
            color=grouped_df["count"],
            size_max=40)


    save_view_fig(fig, image_type, figure_name, analysis_name, view_fig, save_fig, save_fig_path)

    return

def make_bubble_plot(
    table_df,
    image_type="png",
    figure_name="<number-or-name>-bubbleplot",
    analysis_name="analysis-<name>",
    metric="LBGI",
    level_of_analysis= "analysis_type",
    view_fig=True,
    save_fig=True,
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
):
    if level_of_analysis == "all":

        df = table_df[[metric]]
        grouped_df = df.groupby([metric]).size().reset_index(name="count")

        fig = px.scatter(
            x=None,
            y=grouped_df[metric],
            size=grouped_df["count"],
            color=grouped_df["count"],
            size_max=40)

    else:

        df = table_df[[level_of_analysis, metric]]
        grouped_df = df.groupby([level_of_analysis, metric]).size().reset_index(name="count")

        fig = px.scatter(
            x=grouped_df[level_of_analysis],
            y=grouped_df[metric],
            size=grouped_df["count"],
            color=grouped_df["count"],
            size_max=40)


    save_view_fig(fig, image_type, figure_name, analysis_name, view_fig, save_fig, save_fig_path)

    return

def save_view_fig(
        fig,
        image_type="png",
        figure_name="<number-or-name>",
        analysis_name="analysis-<name>",
        view_fig=True,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures"),
):
    if view_fig:
        fig.show()

    file_name = "{}-{}_{}_{}".format(analysis_name, figure_name, utc_string, code_version)

    if save_fig:
        pio.write_image(
            fig=fig, file=os.path.join(save_fig_path, file_name + ".{}".format(image_type)), format=image_type
        )

    return

#Iterate through each metric and analysis_level category shown below and create boxplot
#figure with both log scale and linear scale.
metrics = ["LBGI"]
analysis_levels = ["bg_test_condition", "analysis_type", "all"]
y_axis_scales = ["log", "linear"]

for analysis_level, metric, axis_scale in itertools.product(analysis_levels, metrics, y_axis_scales):
    make_boxplot(
        summary_df_reduced,
        figure_name= "boxplot-" + analysis_level + "-" + metric,
        analysis_name="icgm-sensitivity-analysis",
        metric=metric,
        level_of_analysis=analysis_level,
        notched_boxplot = False,
        y_scale_type = axis_scale,
        image_type="png",
        view_fig=False,
        save_fig=True  # This is not working, need to figure out why
    )


metrics = ["LBGI Risk Score", "DKAI Risk Score"]
analysis_levels = ["bg_test_condition", "analysis_type", "all"]

for analysis_level, metric in itertools.product(analysis_levels, metrics):
    make_bubble_plot(
        summary_df_reduced,
        image_type="png",
        figure_name="bubbleplot-" + analysis_level + "-" + metric,
        analysis_name="icgm-sensitivity-analysis",
        metric=metric,
        level_of_analysis=analysis_level,
        view_fig=True,
        save_fig=True,
        save_fig_path=os.path.join("..", "..", "reports", "figures")
    )



# %% Distribution of Batch Sensor Characteristics
# # filter data by virtual_patient and bg_test_condtion
# sensor_batch = sensor_df.groupby(["virtual_patient_num", "bg_test_condition"])
# sensor_reduced_df = sensor_batch[["sensor_A_results"]].median()
# sensor_reduced_df.reset_index(inplace=True)
#
# # generic boxplot
# fig = px.box(
#     sensor_reduced_df,
#     y="sensor_A_results",
#     points="all",
#     # labels=dict(sensor_MARD="Batch Sensor Noise Coefficient"),
# )
# plot(fig)
#
# # %% Table 5A. iCGM Special Controls Results Example
# table_name = "iCGM-Special-Controls-Results"
# icgm_table_columns = [
#     "Criterion",
#     "iCGM Thresholds<br>95% Lower Confidence Bound*",
#     "Batch Sensor Results<br>95% Lower Confidence Bound*",
#     "Number of iCGM-True Pairs in Criterion",
# ]
# icgm_special_controls_table = pd.DataFrame(columns=icgm_table_columns)
#
# # %% Table 5B6. Batch iCGM Sensor Characteristics Example.
#
# # %% Table 6. Definition of Severity for DKA
#
# # %% Table 7. Definition of Severity for Severe Hypoglycemia
#
# # %% Table 8. Example Summary Table.
#
# # %%
# # COLOR REF: "https://colorbrewer2.org/?type=qualitative&scheme=Set1&n=9"
# color_brewer_colors = [
#     "#e41a1c",
#     "#377eb8",
#     "#4daf4a",
#     "#984ea3",
#     "#ff7f00",
#     "#ffff33",
#     "#a65628",
#     "#f781bf",
#     "#999999",
# ]
