import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from celluloid import Camera
from matplotlib.lines import Line2D

# This script depends on imagemagick (for saving image). You can download imagemagick here:
# http://www.imagemagick.org/script/download.php

#Load in data frame and do data manipulatioin
insulin_pulse_file_location = os.path.join(
    "..", "..", "data", "raw", "2020-06-30_wPyloopkit_Update"
)

filename = 'SBR 0.1 VPBR 0.1 MBR 0.2.csv'

simulation_example_path = os.path.abspath(
    os.path.join(insulin_pulse_file_location, filename)
)

sim_df = pd.read_csv(simulation_example_path)

sim_df["five_minute_marks"] = sim_df.index
sim_df["minutes_post_simulation"] = sim_df["five_minute_marks"].apply(
    lambda x: x * 5
)
sim_df["hours_post_simulation"] = sim_df["minutes_post_simulation"] / 60

#Create figure

fig, axes = plt.subplots(2)
camera = Camera(fig)
t = np.linspace(0, 2 * np.pi, 128, endpoint=False)

axes[0].set_ylabel('BG (mg/dL')
axes[0].set_xlabel('Hours Post Simulation')
axes[0].set_xlim(0, 24)
axes[0].set_ylim(min(sim_df["bg"]-20), max(sim_df["bg"]+20))
axes[0].set_xticks(np.arange(min(sim_df["hours_post_simulation"]), max(sim_df["hours_post_simulation"])+1, 2.0))
axes[0].grid(True)

axes[1].set_ylabel('Insulin (U or U/hr)')
axes[1].set_xlabel('Hours Post Simulation')
axes[1].set_xlim(0, 24)
axes[1].set_ylim(0, 1)
axes[1].set_xticks(np.arange(min(sim_df["hours_post_simulation"]), max(sim_df["hours_post_simulation"])+1, 2.0))
axes[1].grid(True)

for t in sim_df["hours_post_simulation"]:
    sim_df_subset = sim_df[sim_df["hours_post_simulation"]<t] #create subset of the data
    axes[0].plot(sim_df_subset["hours_post_simulation"], sim_df_subset["bg"], color="#D9CEED", linestyle="None", label="True BG", marker='o', markersize=2)
    axes[0].plot(sim_df_subset["hours_post_simulation"], sim_df_subset["bg_sensor"], color="#6AA84F", linestyle='None', label="Sensor BG", marker='o',markersize=2)
    axes[1].plot(sim_df_subset["hours_post_simulation"], sim_df_subset["sbr"], color="#008ECC", linestyle="dotted", label="Scheduled Basal Rate")
    axes[1].plot(sim_df_subset["hours_post_simulation"], sim_df_subset["iob"], color="#271B46", linestyle="solid", label="Insulin on Board")
    axes[1].plot(sim_df_subset["hours_post_simulation"], sim_df_subset["delivered_basal_insulin"], color="#f9706b",linestyle="solid", label="Delivered Basal Insulin")
    camera.snap()

#Create custom legend
labels = ['True BG', 'Sensor BG', 'Scheduled Basal Rate',
          'Insulin on Board','Delivered Basal Insulin']
colors = ["#D9CEED", "#6AA84F", "#008ECC", "#271B46", "#f9706b"]
handles = []
for c, l in zip(colors, labels):
    handles.append(Line2D([0], [0], color = c, label = l))

axes[0].legend(handles = handles, loc = 'upper right')


fig.tight_layout()

animation = camera.animate()

animation.save('animation.gif', writer='imagemagick', fps=30)

plt.show()



#animation.save('celluloid_subplots.gif') #, writer='imagemagick')