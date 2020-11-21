__author__ = "Anne Evered"

# %% REQUIRED LIBRARIES
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from celluloid import Camera
from matplotlib.lines import Line2D
import datetime as dt

# This script depends on imagemagick (for saving image). You can download imagemagick here:
# http://www.imagemagick.org/script/download.php


# Create figure
def create_insulin_pulse_animation(file_location, filename):
    """

    Creates an animation in matplotlib to visualize missed insulin pulses analysis.

    Parameters
    ----------
    file_location: str
        location of files to show in animation
    filename: str
        name of file to show in animation

    Returns
    -------

    """

    # Load in data
    simulation_example_path = os.path.abspath(os.path.join(file_location, filename))

    sim_df = pd.read_csv(simulation_example_path)

    # Additional data preparation
    sim_df["five_minute_marks"] = sim_df.index
    sim_df["minutes_post_simulation"] = sim_df["five_minute_marks"].apply(
        lambda x: x * 5
    )
    sim_df["hours_post_simulation"] = sim_df["minutes_post_simulation"] / 60

    sim_df["temp_basal_sbr_if_nan"] = sim_df["temp_basal"].mask(
        pd.isnull, sim_df["sbr"]
    )

    # Set fonts
    font = {"size": 8}
    plt.rc("font", **font)

    fig, axes = plt.subplots(3)
    camera = Camera(fig)

    fig.set_size_inches(10, 6)

    # Parse out MBR from filename
    start = "MBR "
    end = ".csv"
    mbr_string = (filename.split(start))[1].split(end)[0]

    # Add layout features
    axes[0].set_ylabel("Glucose (mg/dL)")
    axes[0].set_xlabel("Hours")
    axes[0].set_xlim(0, 24)
    axes[0].set_ylim(min(sim_df["bg"] - 10), max(sim_df["bg"] + 10))
    axes[0].set_xticks(
        np.arange(
            min(sim_df["hours_post_simulation"]),
            max(sim_df["hours_post_simulation"]) + 1,
            2.0,
        )
    )
    axes[0].grid(True)
    axes[0].set_title(
        "Simulation where scheduled basal rate is "
        + str(max(sim_df["sbr"]))
        + " and maximum allowable temp basal is "
        + mbr_string,
        fontsize=9,
    )

    axes[1].set_ylabel("Insulin (U or U/hr)")
    axes[1].set_xlabel("Hours")
    axes[1].set_xlim(0, 24)
    axes[1].set_ylim(
        0,
        max(
            max(sim_df["temp_basal"].fillna(sim_df["sbr"].max())) + 0.05,
            max(sim_df["iob"]) + 0.05,
        ),
    )
    axes[1].set_xticks(
        np.arange(
            min(sim_df["hours_post_simulation"]),
            max(sim_df["hours_post_simulation"]) + 1,
            2.0,
        )
    )
    axes[1].grid(True)

    axes[2].set_ylabel("Insulin (U or U/hr)")
    axes[2].set_xlabel("Hours")
    axes[2].set_xlim(0, 24)
    axes[2].set_ylim(0, max(sim_df["temp_basal"].fillna(sim_df["sbr"].max())) + 0.05)
    axes[2].set_xticks(
        np.arange(
            min(sim_df["hours_post_simulation"]),
            max(sim_df["hours_post_simulation"]) + 1,
            2.0,
        )
    )
    axes[2].grid(True)
    axes[2].set_title(
        "Loop Decisions (scheduled basal rate and temp basals)", fontsize=8
    )

    # Add in different animation traces
    for t in sim_df["hours_post_simulation"]:
        sim_df_subset = sim_df[
            sim_df["hours_post_simulation"] < t
        ]  # create subset of the data
        axes[0].plot(
            sim_df_subset["hours_post_simulation"],
            sim_df_subset["bg"],
            color="#B1BEFF",
            linestyle="solid",
            marker="o",
            markersize=2,
        )
        axes[0].plot(
            sim_df_subset["hours_post_simulation"],
            sim_df_subset["bg_sensor"],
            color="#6AA84F",
            alpha=0.5,
            linestyle="solid",
            marker="o",
            markersize=2,
        )
        axes[1].plot(
            sim_df_subset["hours_post_simulation"],
            sim_df_subset["sbr"] / 2,
            color="black",
            linewidth=1,
            linestyle="--",
        )
        axes[1].plot(
            sim_df_subset["hours_post_simulation"],
            sim_df_subset["iob"],
            color="#744AC2",
            linestyle="solid",
        )
        (markerLines, stemLines, baseLines) = axes[1].stem(
            sim_df_subset["hours_post_simulation"],
            sim_df_subset["delivered_basal_insulin"],
            linefmt="#f9706b",
            use_line_collection=True,
        )
        plt.setp(markerLines, color="#f9706b", markersize=2)
        plt.setp(stemLines, color="#f9706b", linewidth=1)
        plt.setp(baseLines, color="#f9706b", linewidth=1)

        axes[2].plot(
            sim_df_subset["hours_post_simulation"],
            sim_df_subset["sbr"],
            color="black",
            linewidth=1,
            linestyle="--",
        )

        axes[2].fill_between(
            sim_df_subset["hours_post_simulation"],
            sim_df_subset["temp_basal_sbr_if_nan"],
            color="#008ECC",
            step="pre",
            alpha=0.4,
        )

        axes[2].plot(
            sim_df_subset["hours_post_simulation"],
            sim_df_subset["temp_basal_sbr_if_nan"],
            color="#008ECC",
            drawstyle="steps-pre",
            linewidth=2,
        )

        camera.snap()
    # %%
    # Create custom legend
    bg_labels = ["True BG", "Sensor EGV"]
    bg_colors = ["#B1BEFF", "#6AA84F"]

    insulin_labels = [
        "50% of Scheduled Basal Rate",
        "Insulin on Board",
        "Delivered Basal Insulin Pulse",
    ]  # , 'Temp Basal']
    insulin_colors = ["black", "#744AC2", "#F9706B"]  # , "#008ECC"]
    insulin_line_styles = ["--", "-", "-"]  # ,'-']

    bg_handles = []
    insulin_handles = []
    loop_decision_handles = []

    for c, l in zip(bg_colors, bg_labels):
        bg_handles.append(
            Line2D(
                [0], [0], color=c, label=l, marker="o", markersize=3, linestyle="None"
            )
        )

    for c, l, s in zip(insulin_colors, insulin_labels, insulin_line_styles):
        insulin_handles.append(
            Line2D([0], [0], color=c, label=l, linestyle=s, linewidth=1.5)
        )

    loop_decision_handles.append(
        Line2D(
            [0], [0], color="#008ECC", label="Loop Decision", linestyle="-", linewidth=1
        )
    )
    loop_decision_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            label="Scheduled Basal Rate",
            linestyle="--",
            linewidth=1,
        )
    )

    axes[0].legend(handles=bg_handles, loc="upper right")
    axes[1].legend(handles=insulin_handles, loc="upper right")
    axes[2].legend(handles=loop_decision_handles, loc="upper right")

    # Set layout
    fig.tight_layout()

    # Add animation
    animation = camera.animate()

    # Save and plot figure
    # plt.show()

    utc_string = dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%m-%S")
    code_version = "v0-1-0"

    animation_file_name = "{}-{}_{}_{}_{}".format(
        "insulin-pulses-analysis", "animation", filename, utc_string, code_version
    )

    # Path to save figure at
    save_fig_path = os.path.join(
        "..",
        "..",
        "reports",
        "figures",
        "insulin-pulses-risk-assessment",
        "animations"
    )

    # Make path if it doesn't exist yet
    if not os.path.exists(save_fig_path):
        print("making directory " + save_fig_path + "...")
        os.makedirs(save_fig_path)

    animation.save(os.path.join(save_fig_path, animation_file_name + ".gif"), writer="imagemagick", fps=5, dpi=100)


# TODO: split above into additional functions
# TODO: add in parameters for saving and viewing fig
# ^ Note: this has been done in generalized version of this animation code
# simulation_figure_matplotlib.py. This is older code that predates simulation_figure_matplotlib.py.


if __name__ == '__main__':

    # Create Figures
    insulin_pulse_file_location = os.path.join(
        "..", "..", "data", "raw", "insulin-pulses-sample-files-2020-07-02"
    )

    filename = "2020-07-02-06-41-20-SBR 0.1 VPBR 0.1 MBR 0.2.csv"
    create_insulin_pulse_animation(insulin_pulse_file_location, filename)

    filename = "2020-07-02-06-41-20-SBR 0.05 VPBR 0.05 MBR 0.1.csv"
    create_insulin_pulse_animation(insulin_pulse_file_location, filename)

    filename = "2020-07-02-06-41-20-SBR 0.05 VPBR 0.05 MBR 0.25.csv"
    create_insulin_pulse_animation(insulin_pulse_file_location, filename)
