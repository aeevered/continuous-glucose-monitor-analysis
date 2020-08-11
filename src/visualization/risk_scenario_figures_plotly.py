import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import os
from save_view_fig import save_view_fig, save_animation
from risk_scenario_figures_shared_functions import (
    data_loading_and_preparation,
    get_features_dictionary,
)


# reference: https://chart-studio.plotly.com/~empet/15243/animating-traces-in-subplotsbr/#/

def add_plot(fig, df, field, row):
    features_dictionary = get_features_dictionary(field)

    if field in [
        "true_bolus",
        "reported_bolus",
        "true_carb_value",
        "reported_carb_value",
        "true_carb_duration",
        "reported_carb_duration",
        "undelivered_basal_insulin",
    ]:
        # filter out the zero values
        df = df[df[field] != 0]

    fig.add_trace(
        go.Scatter(
            x=df["hours_post_simulation"],
            y=df[field],
            hoverinfo="y",
            yaxis="y" + str(row),
            line_shape=features_dictionary["shape"],
            line_dash=features_dictionary["dash"],
            mode=features_dictionary["mode"],
            legendgroup=features_dictionary["legend_label"],
            line_color=features_dictionary["color"],
            name=features_dictionary["legend_label"],
            showlegend=True,
            fill=features_dictionary["fill"],
        ),
        row=row,
        col=1,
    )

    return fig


def set_layout(
    traces, num_subplots, fig, data_frames, time_range=(0, 8)
):
    # fig.update_layout(width=700, height=475)
    fig.update_xaxes(range=[time_range[0], time_range[1]], title="Hours")

    for subplot in range(num_subplots):
        # Get min and max value
        fields_in_subplot = []
        min_value = 100
        max_value = 0
        for index, traces_per_file in enumerate(traces):
            if subplot in traces_per_file:
                fields = traces_per_file[subplot]
            else:
                fields = []
            fields_in_subplot = fields_in_subplot + fields
            for field in fields:
                if max(data_frames[index][field]) > max_value:
                    max_value = max(data_frames[index][field])
                if min(data_frames[index][field]) < min_value:
                    min_value = min(data_frames[index][field])

        # TODO: add in the title features here
        if "bg" in fields_in_subplot or "bg_sensor" in fields_in_subplot:
            fig.update_yaxes(
                range=[min_value - 20, max_value + 50],
                title="Glucose (mg/dL)",
                row=subplot + 1,
                col=1,
            )
        else:
            fig.update_yaxes(
                range=[0, max_value + 0.5],
                title="Insulin (U or U/hr)",
                row=subplot + 1,
                col=1,
            )
    return fig


# Create figure
def create_simulation_figure_plotly(
    file_location,
    file_names,
    traces,
    subplots,
    time_range=(0, 8),
    main_title="Risk Scenario Simulation",
    subplot_titles=[],
    save_fig_path="",
    figure_name="simulation_figure",
    analysis_name="risk-scenarios",
    animate=True,
):
    # Load data files
    data_frames = []
    for file in file_names:
        df = data_loading_and_preparation(
            os.path.abspath(os.path.join(file_location, file))
        )
        data_frames.append(df)

    # Set up figure and axes
    print(subplot_titles)
    fig = make_subplots(rows=subplots, cols=1, subplot_titles=np.array(subplot_titles))

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=14)

    fig = set_layout(traces, subplots, fig, data_frames, time_range)

    # Add plots
    for index, df in enumerate(data_frames):
        for subplot in traces[index]:
            for field in traces[index][subplot]:
                row = subplot + 1
                fig = add_plot(fig, df, field, row)

    # add annotations
    fig["layout"]["annotations"] += (
        dict(
            xref="paper",
            yref="paper",
            x=0.55,
            y=-0.27,
            showarrow=False,
            text="This is a caption for the Plotly figure. It could be a description of the "
                 "risk scenario or the particular details of the scenario.",
            font=dict(size=13)
        ),
        dict(
            xref="paper",
            yref="paper",
            x=1.08,
            y=.1,
            showarrow=False,
            text="SBR = 0.3",
        ),
    )

    if animate:
        time_chunks = list(np.arange(time_range[0], time_range[1] + 0.2, 0.25))
        num_frames = len(time_chunks)

        frames = []

        for k in range(num_frames):
            data = []
            num_traces = 0
            for index, df in enumerate(data_frames):
                for subplot in traces[index]:
                    for field in traces[index][subplot]:
                        data.extend(
                            [
                                go.Scatter(
                                    y=df[df["hours_post_simulation"] <= time_chunks[k]][
                                        field
                                    ]
                                )
                            ]
                        )
                        num_traces += 1
            frames.append(
                dict(
                    name=k,
                    data=data,
                    # update the trace 1 in (1,1)
                    traces=list(np.arange(num_traces))
                    # [0, 1, 2]  # the elements of the list [0,1,2] give info on the traces in fig.data
                    # that are updated by the above three go.Scatter instances
                )
            )

        updatemenus = [
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label=">",
                        method="animate",
                        args=[
                            [f"{k}" for k in range(num_frames)],
                            dict(
                                frame=dict(duration=500, redraw=False),
                                transition=dict(duration=0),
                                easing="linear",
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="=",
                        method="animate",
                        args=[
                            [],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
                direction="left",
                pad=dict(r=10, t=85),
                showactive=False,
                x=0.08,
                y=-0.05,
                xanchor="right",
                yanchor="top",
            )
        ]

        sliders = [
            {
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Frame: ",
                    "visible": False,
                    "xanchor": "right",
                },
                "transition": {"duration": 500.0, "easing": "linear"},
                "pad": {"b": 10, "t": 50},
                "len": 1,
                "x": 0,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [k],
                            {
                                "frame": {
                                    "duration": 500.0,
                                    "easing": "linear",
                                    "redraw": False,
                                },
                                "transition": {"duration": 0, "easing": "linear"},
                            },
                        ],
                        "label": time_chunks[k],
                        "method": "animate",
                    }
                    for k in range(num_frames)
                ],
            }
        ]

        fig.update(frames=frames),
        fig.update_layout(
            title_text=main_title,
            title_font_size=16,
            margin_b=150,
            margin_t=90,
            updatemenus=updatemenus,
            sliders=sliders,
            font=dict(
                size=10
            )
        )

    else:
        fig.update_layout(
            title_text=main_title,
            title_font_size=16,
            margin_b=150,
            margin_t=90,
            font=dict(
                size=10
            )
        )

    save_view_fig(
        fig,
        image_type="png",
        figure_name=figure_name,
        analysis_name=analysis_name,
        view_fig=True,
        save_fig=False,
        save_fig_path=save_fig_path,
        width=600,
        height=700,
    )

    return


file_location = os.path.join("..", "..", "data", "processed")
loop_filename = "risk_scenarios_PyLoopkit v0.1.csv"
no_loop_filename = "risk_scenarios_do_nothing.csv"

traces = [{0: ["bg", "bg_sensor"], 1: ["iob"], 2: ["sbr"]}]

create_simulation_figure_plotly(
    file_location=file_location,
    file_names=[no_loop_filename],
    traces=traces,
    subplots=3,
    time_range=(0, 8),
    main_title="Risk Scenario",
    subplot_titles = [
                     "BG Values",
                     "Insulin On-Board",
                     "Scheduled Basal Rate",
                 ],
    save_fig_path=os.path.join("..", "..", "reports", "figures", "fda-risk-scenarios"),
    figure_name="plotly_simulation_figure",
    analysis_name="risk_scenarios",
    animate=True,
)

traces = [{0: ["bg", "bg_sensor"], 1: ["iob"], 2: ["sbr", "temp_basal_sbr_if_nan"]}]

create_simulation_figure_plotly(
    file_location=file_location,
    file_names=[loop_filename],
    traces=traces,
    subplots=3,
    time_range=(0, 8),
    main_title="Risk Scenario",
    subplot_titles=[
        "BG Values",
        "Insulin On-Board",
        "Scheduled Basal Rate and Loop Decisions",
    ],
    save_fig_path=os.path.join("..", "..", "reports", "figures", "fda-risk-scenarios"),
    figure_name="plotly_simulation_figure",
    analysis_name="risk_scenarios",
    animate=True,
)
