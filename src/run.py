import argparse
import config
from dash import Dash, html, dcc, page_container, Input, Output
import dash_bootstrap_components as dbc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default=config.DATA_DIR, help="Path to data directory"
    )
    args = parser.parse_args()
    config.DATA_DIR = args.data_dir

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],  # bootstrap-icons served locally from assets/
)

app.layout = html.Div(
    children=[
        dcc.Location(
            id="url", refresh=False
        ),  # track the URL and switch between pages based on tab selection
        dcc.Store(id="data-path-store", storage_type="session"),
        dcc.Store(id="chunk-limits-store", data=[], storage_type="session"),
        dcc.Store(id="frequency-store", storage_type="session"),
        dcc.Store(id="annotation-store", data=[], storage_type="session"),
        dcc.Store(id="channel-store", data={}, storage_type="session"),
        dcc.Store(id="display-channel-store", data={}, storage_type="session"),
        dcc.Store(id="montage-store", data={}, storage_type="local"),
        dcc.Store(id="history-store", data={}, storage_type="session"),
        dcc.Store(id="model-probabilities-store", data={}, storage_type="session"),
        dcc.Store(id="sensitivity-analysis-store", data={}, storage_type="session"),
        dcc.Store(id="ica-store", data=[], storage_type="session"),
        dcc.Store(id="raw-modality", storage_type="session"),
        dcc.Store(id="pending-has-scalp-eeg", storage_type="session"),
        dcc.Store(id="ica-components-dir-store", storage_type="session"),
        dcc.Store(id="model-csv-store", data={}, storage_type="session"),
        html.Div(
            children=[
                html.Div(
                    id="header-bandeau",
                    children=[
                        # left section: Logo and navigation menu
                        html.Div(
                            children=[
                                dbc.DropdownMenu(
                                    children=[
                                        dbc.DropdownMenuItem("Dataset", header=True),
                                        dbc.DropdownMenuItem("Home", href="/"),
                                        dbc.DropdownMenuItem(divider=True),
                                        dbc.DropdownMenuItem(
                                            "Visualization", header=True
                                        ),
                                        dbc.DropdownMenuItem(
                                            "Raw Signal", href="/viz/raw-signal"
                                        ),
                                        dbc.DropdownMenuItem("ICA", href="/viz/ica"),
                                        dbc.DropdownMenuItem(divider=True),
                                        dbc.DropdownMenuItem("Model", header=True),
                                        dbc.DropdownMenuItem(
                                            "Custom", href="/model/custom"
                                        ),
                                        dbc.DropdownMenuItem(
                                            "Performance", href="/model/performance"
                                        ),
                                        dbc.DropdownMenuItem(
                                            "Fine-Tuning", href="/model/fine-tuning"
                                        ),
                                        dbc.DropdownMenuItem(divider=True),
                                        dbc.DropdownMenuItem("Settings", header=True),
                                        dbc.DropdownMenuItem(
                                            "Montage", href="/settings/montage"
                                        ),
                                        dbc.DropdownMenuItem(
                                            "Help", href="/settings/help"
                                        ),
                                    ],
                                    toggle_class_name="bi bi-list",
                                    toggle_style={
                                        "fontSize": "50px",
                                        "cursor": "pointer",
                                    },
                                    direction="down",
                                    in_navbar=True,
                                    nav=True,
                                    caret=False,
                                ),
                                html.Img(
                                    src="/assets/deepepix-logo.jpeg",
                                    style={"borderRadius": "10%", "height": "36px"},
                                ),
                                html.Span(
                                    "DeepEpiX ©",
                                    style={
                                        "fontWeight": "bold",
                                        "fontSize": "16px",
                                        "marginLeft": "8px",
                                        "alignSelf": "center",
                                    },
                                ),
                            ],
                            style={
                                "marginLeft": "20px",
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "30px",
                            },
                        ),
                        # right section: Logos
                        html.Div(
                            children=[
                                html.Span(
                                    [
                                        dbc.Label(
                                            className="bi bi-moon",
                                            html_for="color-mode-switch",
                                        ),
                                        dbc.Switch(
                                            id="color-mode-switch",
                                            value=True,
                                            className="d-inline-block ms-1",
                                            persistence=True,
                                        ),
                                        dbc.Label(
                                            className="bi bi-sun",
                                            html_for="color-mode-switch",
                                        ),
                                    ]
                                ),
                                dcc.Store(id="theme-store"),
                                html.Img(
                                    src="/assets/crnl-logo.png",
                                    style={
                                        "height": "36px",
                                        "borderRadius": "20%",
                                        "padding": "3px",
                                        "backgroundColor": "#ffffff",  # Set any hex color or name
                                    },
                                ),
                                html.Img(
                                    src="/assets/inserm-logo.png",
                                    style={
                                        "height": "36px",
                                        "borderRadius": "10%",
                                        "padding": "8px",
                                        "backgroundColor": "#ffffff",  # Set any hex color or name
                                    },
                                ),
                                html.Img(
                                    src="/assets/ucbl-logo.jpg",
                                    style={
                                        "height": "36px",
                                        "borderRadius": "15%",
                                        "padding": "6px",
                                        "marginRight": "20px",
                                        "backgroundColor": "#ffffff",  # Set any hex color or name
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "20px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "zIndex": "1200",
                        "background": "linear-gradient(90deg, rgba(13, 110, 253, 0.7), rgba(255, 105, 180, 0.7))",
                        "position": "fixed",
                        "top": 0,
                        "left": 0,
                        "width": "100%",
                    },
                ),
                # main content container
                html.Div(
                    children=[
                        page_container,
                    ],
                    style={
                        "padding-top": "80px",
                        "width": "98%",
                        "margin": "0 auto",
                        "display": "inline-block",
                    },
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",
            },
        ),
    ]
)

# callback to switch light/dark mode
app.clientside_callback(
    """
    function(dark_mode) {
        document.documentElement.setAttribute('data-bs-theme', dark_mode ? 'light' : 'dark');
        return dark_mode ? 'light' : 'dark';
    }
    """,
    Output("theme-store", "data"),
    Input("color-mode-switch", "value"),
)

server = app.server

if __name__ == "__main__":

    app.run(debug=config.DEBUG, port=config.PORT)
