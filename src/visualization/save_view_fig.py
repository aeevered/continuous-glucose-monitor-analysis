# %% REQUIRED LIBRARIES
import os
import plotly.io as pio
import datetime as dt


def save_view_fig(
    fig,
    image_type="png",
    figure_name="<number-or-name>",
    analysis_name="analysis-<name>",
    view_fig=True,
    save_fig=True,
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
    width=600,
    height=700,
):
    if view_fig:
        fig.show()

    utc_string = dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%m-%S")
    code_version = "v0-1-0"

    file_name = "{}-{}_{}_{}".format(
        analysis_name, figure_name, utc_string, code_version
    )

    if save_fig:
        pio.write_image(
            fig=fig,
            file=os.path.join(save_fig_path, file_name + ".{}".format(image_type)),
            format=image_type,
            width=width,
            height=height,
        )

    return
