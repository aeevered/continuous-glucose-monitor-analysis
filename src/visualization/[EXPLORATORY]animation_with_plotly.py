# %% REQUIRED LIBRARIES
import pandas as pd
import os

import plotly.express as px

insulin_pulse_file_location = os.path.join(
    "..", "..", "data", "raw", "2020-07-02"
)

filename = "2020-07-02-06-41-20-SBR 0.1 VPBR 0.1 MBR 0.2.csv"

simulation_example_path = os.path.abspath(
    os.path.join(insulin_pulse_file_location, filename)
)

sim_df = pd.read_csv(simulation_example_path)

sim_df["five_minute_marks"] = sim_df.index
sim_df["minutes_post_simulation"] = sim_df["five_minute_marks"].apply(lambda x: x * 5)
sim_df["hours_post_simulation"] = sim_df["minutes_post_simulation"] / 60

fig = px.scatter(
    sim_df,
    x="hours_post_simulation",
    y="bg",
    animation_frame="hours_post_simulation",  # animation_group="country",
    size="bg",
    color="bg",
    hover_name="bg",
    size_max=4,
    range_x=[0, 24],
    range_y=[1, 250],
)

fig.show()

color_dict = {
    "bg": "purple",
    "bg_sensor": "red",
    "sbr": "yellow",
    "iob": "blue",
    "delivered_basal_insulin": "orange",
}

# Group the metrics into categories into column metric_type
sim_df = pd.melt(
    sim_df,
    id_vars=["hours_post_simulation"],
    var_name="metric",
    value_name="metric_value",
)


metrics_mapping = {
    "bg": "bg",
    "bg_sensor": "bg",
    "sbr": "insulin",
    "iob": "insulin",
    "delivered_basal_insulin": "insulin",
}

sim_df = sim_df[sim_df["metric"].isin(metrics_mapping.keys())]

sim_df["metric_type"] = sim_df["metric"].map(metrics_mapping)


fig = px.scatter(
    sim_df,
    x="hours_post_simulation",
    y="metric_value",
    animation_frame="hours_post_simulation",
    animation_group="metric",
    color="metric",
    color_discrete_map=color_dict,
    hover_name="metric_value",
    size_max=4,
    range_x=[0, 24],
    range_y=[0, 250],
)

fig.show()


fig = px.scatter(
    sim_df,
    x="hours_post_simulation",
    y="metric_value",
    animation_frame="hours_post_simulation",  # animation_group="country",
    color="metric",
    hover_name="metric_value",
    facet_row="metric_type",  # size="bg",
    size_max=30,
    range_x=[0, 24],
    color_discrete_map=color_dict,
)  # , range_y=[0,250]

fig.show()
