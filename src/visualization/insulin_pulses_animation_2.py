import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from celluloid import Camera
from matplotlib.lines import Line2D
import re

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


#Parse out MBR from filename
start = 'MBR '
end = '.csv'
mbr_string = (filename.split(start))[1].split(end)[0]

#Set fonts
font = {'size': 8}
plt.rc('font', **font)

#Create figure
fig, axes = plt.subplots(2)
camera = Camera(fig)


axes[0].set_ylabel('Glucose (mg/dL)')
axes[0].set_xlabel('Hours')
axes[0].set_xlim(0, 24)
axes[0].set_ylim(min(sim_df["bg"]-20), max(sim_df["bg"]+40))
axes[0].set_xticks(np.arange(min(sim_df["hours_post_simulation"]), max(sim_df["hours_post_simulation"])+1, 2.0))
axes[0].grid(True)
axes[0].set_title("Simulation where scheduled basal rate is " + str(max(sim_df["sbr"])) +" and maximum allowable temp basal is " + mbr_string, fontsize=8)


axes[1].set_ylabel('Insulin (U or U/hr)')
axes[1].set_xlabel('Hours')
axes[1].set_xlim(0, 24)
axes[1].set_ylim(0, max(sim_df["iob"])+.2)
axes[1].set_xticks(np.arange(min(sim_df["hours_post_simulation"]), max(sim_df["hours_post_simulation"])+1, 2.0))
axes[1].grid(True)


for t in sim_df["hours_post_simulation"]:
    sim_df_subset = sim_df[sim_df["hours_post_simulation"]<t] #create subset of the data
    axes[0].plot(sim_df_subset["hours_post_simulation"], sim_df_subset["bg"], color="#B1BEFF", linestyle="None", marker='o', markersize=2)
    axes[0].plot(sim_df_subset["hours_post_simulation"], sim_df_subset["bg_sensor"], color="#6AA84F", alpha=0.8, linestyle='None', marker='o', markersize=2)
    axes[1].plot(sim_df_subset["hours_post_simulation"], sim_df_subset["sbr"], color="#008ECC", linewidth=2, linestyle="--")
    axes[1].plot(sim_df_subset["hours_post_simulation"], sim_df_subset["iob"], color="#744AC2", linestyle="solid")
    (markerLines, stemLines, baseLines) = axes[1].stem(sim_df_subset["hours_post_simulation"], sim_df_subset["delivered_basal_insulin"], linefmt="#f9706b")
    plt.setp(markerLines, color="#f9706b", markersize=2)
    plt.setp(stemLines, color="#f9706b", linewidth=1)
    plt.setp(baseLines, color="#f9706b", linewidth=1)

    axes[1].plot(sim_df_subset["hours_post_simulation"], sim_df_subset["temp_basal"], color="#008ECC", drawstyle="steps-pre")

    camera.snap()

#Create custom legend
bg_labels = ['True BG', 'Sensor Glucose']
bg_colors = ["#B1BEFF", "#6AA84F"]

insulin_labels = ['Scheduled Basal Rate','Insulin on Board','Delivered Basal Insulin', 'Temp Basal']
insulin_colors = ["#008ECC","#744AC2", "#F9706B", "#008ECC"]
insulin_line_styles = ['--','-','-','-']

bg_handles = []
insulin_handles = []

for c, l in zip(bg_colors, bg_labels):
    bg_handles.append(Line2D([0], [0], color=c, label=l, marker='o', markersize=3, linestyle="None"))

for c, l, s in zip(insulin_colors, insulin_labels, insulin_line_styles):
    insulin_handles.append(Line2D([0], [0], color=c, label=l, linestyle=s, linewidth=1.5))

axes[0].legend(handles=bg_handles, loc='upper right')
axes[1].legend(handles=insulin_handles, loc='upper right')


#Plot figure and add animation
fig.tight_layout()

animation = camera.animate()

animation.save('animation.gif', writer='imagemagick', fps=30)

plt.show()

#animation.save('celluloid_subplots.gif') #, writer='imagemagick')