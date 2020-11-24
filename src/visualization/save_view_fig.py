__author__ = "Anne Evered"

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
    code_version="v0-1-0",
):
    """
    Takes a plotly figure object and shows and/or saves that figure.

    Parameters
    ----------
    fig: plotly figure object
        the plotly figure (or dict representing a figure) to save
    image_type: str
        the file type to save image as ("jpg","jpeg","png","webp","svg","pdf")
    figure_name: str
        the name to use for the figure
    analysis_name: str
        the name of the analysis this figure is a part of
    view_fig: bool
        whether or not to view the figure (launches in web browser)
    save_fig: bool
        whether or not to save the figure
    save_fig_path: str
        the file location to store the figure at
    width: int
        figure width in pixels
    height: int
        figure height in pixels
    code_version: str
        what version code is for the purpose of versioning in figure name

    Returns
    -------

    """

    if view_fig:
        fig.show()

    # Get datetime
    utc_string = dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%m-%S")

    # Create filename from function parameters and date string
    file_name = "{}-{}_{}_{}".format(
        analysis_name, figure_name, utc_string, code_version
    )

    if save_fig:

        # Create directory if it does not exist
        if not os.path.exists(save_fig_path):
            print("making directory " + save_fig_path + "...")
            os.makedirs(save_fig_path)

        # Write image
        pio.write_image(
            fig=fig,
            file=os.path.join(save_fig_path, file_name + ".{}".format(image_type)),
            format=image_type,
            width=width,
            height=height,
            scale=2,
        )

    return


def save_animation(
    animation,
    figure_name="<number-or-name>",
    analysis_name="analysis-<name>",
    save_fig=True,
    save_fig_path=os.path.join("..", "..", "reports", "figures"),
    fps=5,
    dpi=100,
    code_version="v0-1-0",
):
    """
    Takes a matplotlib figure and saves that figure as an animation using imagemagick.

    Parameters
    ----------
    animation: figure object
        matplotlib figure to animate
    figure_name: str
        the name to use for the figure
    analysis_name: str
        the name of the analysis this figure is part of (used in overall figure name)
    save_fig: bool
        whether or not to save the figure
    save_fig_path: str
        the file location to store the figure at
    fps: int
        frames per second
    dpi: int
        controls the dots per inch; impacts animation size
    code_version
        what version code is for the purpose of versioning in figure name

    Returns
    -------

    """

    # Get datetime
    utc_string = dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%m-%S")

    # Create filename from function parameters and date string
    file_name = "{}-{}_{}_{}".format(
        analysis_name, figure_name, utc_string, code_version
    )

    # If save_fig is True, create directory and save figure as animation
    if save_fig:

        # Create directory if it does not exist
        if not os.path.exists(save_fig_path):
            print("making directory " + save_fig_path + "...")
            os.makedirs(save_fig_path)

        # Save animation
        animation.save(
            os.path.join(save_fig_path, file_name + ".gif"),
            writer="imagemagick",
            fps=fps,
            dpi=dpi,
        )

    return
