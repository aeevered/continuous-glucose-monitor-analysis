import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
import os

fig_path = os.path.join(
    "..", "..", "reports", "figures", "icgm-sensitivity-paired-comparison-figures"
)
combined_df = pd.read_csv(
    "/Users/anneevered/Desktop/Tidepool 2020/Tidepool Repositories/data-science--explore--risk-sim-figures/reports/figures/pairwise_comparison_combined_df_icgm-sensitivity-analysis-results-2020-10-12-nogit.csv"
)

combined_df["sensor_num_icgm"] = combined_df["sensor_num_icgm"].apply(lambda x: int(x))
combined_df = combined_df.sort_values(by=["sensor_num_icgm"])

combined_df["sensor_num_icgm_string"] = combined_df["sensor_num_icgm"].apply(
    lambda x: "Sensor " + str(x)
)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# make a sample data frame with 6 columns
np.random.seed(0)
# df = pd.DataFrame({"Col " + str(i+1): np.random.rand(30) for i in range(6)})

app.layout = html.Div(
    [
        html.Div(
            dcc.Graph(id="g1", config={"displayModeBar": False}),
            className="four columns",
        ),
        html.Div(
            dcc.Graph(id="g2", config={"displayModeBar": False}),
            className="four columns",
        ),
        html.Div(
            dcc.Graph(id="g3", config={"displayModeBar": False}),
            className="four columns",
        ),
    ],
    className="row",
)


def get_figure(df, x_col, y_col, selectedpoints, selectedpoints_local):

    if selectedpoints_local and selectedpoints_local["range"]:
        ranges = selectedpoints_local["range"]
        selection_bounds = {
            "x0": ranges["x"][0],
            "x1": ranges["x"][1],
            "y0": ranges["y"][0],
            "y1": ranges["y"][1],
        }
    else:
        selection_bounds = {
            "x0": np.min(df[x_col]),
            "x1": np.max(df[x_col]),
            "y0": np.min(df[y_col]),
            "y1": np.max(df[y_col]),
        }

    # set which points are selected with the `selectedpoints` property
    # and style those points with the `selected` and `unselected`
    # attribute. see
    # https://medium.com/@plotlygraphs/notes-from-the-latest-plotly-js-release-b035a5b43e21
    # for an explanation
    fig = px.scatter(df, x=df[x_col], y=df[y_col], text=df.index)

    fig.update_traces(
        selectedpoints=selectedpoints,
        customdata=df.index,
        mode="markers+text",
        marker={"color": "rgba(0, 116, 217, 0.7)", "size": 5},
        unselected={
            "marker": {"color": "red", "opacity": 0.3},
            "textfont": {"color": "rgba(0, 0, 0, 0)"},
        },
    )

    fig.update_layout(
        margin={"l": 20, "r": 0, "b": 15, "t": 5}, dragmode="select", hovermode="x"
    )

    fig.add_shape(
        dict(
            {"type": "rect", "line": {"width": 1, "dash": "dot", "color": "darkgrey"}},
            **selection_bounds
        )
    )
    return fig


# this callback defines 3 figures
# as a function of the intersection of their 3 selections
@app.callback(
    [Output("g1", "figure"), Output("g2", "figure"), Output("g3", "figure")],
    [
        Input("g1", "selectedData"),
        Input("g2", "selectedData"),
        Input("g3", "selectedData"),
    ],
)
def callback(selection1, selection2, selection3):
    selectedpoints = combined_df.index
    for selected_data in [selection1, selection2, selection3]:
        if selected_data and selected_data["points"]:
            selectedpoints = np.intersect1d(
                selectedpoints, [p["customdata"] for p in selected_data["points"]]
            )

    return [
        get_figure(
            combined_df,
            "initial_bias_icgm",
            "LBGI Difference",
            selectedpoints,
            selection1,
        ),
        get_figure(
            combined_df,
            "noise_coefficient_icgm",
            "LBGI Difference",
            selectedpoints,
            selection2,
        ),
        get_figure(
            combined_df,
            "bias_drift_range_start_icgm",
            "LBGI Difference",
            selectedpoints,
            selection3,
        ),
    ]


if __name__ == "__main__":
    app.run_server(debug=True)
