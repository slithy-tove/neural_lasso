"""
Inspect all of the trained models in the "results" directory.
"""
import os
import glob
import msgpack
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
from flask import send_from_directory

# generate a list called "headers" of all the files in the directory "results/headers"
header_files = glob.glob(os.path.join("results", "headers", "*.msgpack"))
headers = []
for fp in header_files:
    with open(fp, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=False)
    headers.append(pd.DataFrame([data]))

dfs = []
for h in headers:
    dfs.append(h)

# join all the dataframes in dfs into a single dataframe called df, filling in with NaNs if not all dataframes have the same columns
df = pd.concat(dfs, ignore_index=True, sort=False)

# make a list of the unique values that each column takes (not including nan)
unique_vals = {col: df[col].dropna().unique().tolist() for col in df.columns}

# sort by date descending if a date column exists
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(by="date", ascending=False).reset_index(drop=True)

app = Dash(__name__)

# dropdowns for each column
dropdowns = [
    dcc.Dropdown(
        options=[{"label": str(v), "value": v} for v in unique_vals[col]],
        placeholder=f"Filter {col}",
        id={"type": "filter-dropdown", "index": col},
        clearable=True,
    )
    for col in df.columns
]

app.layout = html.Div(
    [
        html.H2("Model Headers"),
        html.Div(dropdowns, style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
        dash_table.DataTable(
            id="table",
            columns=[{"name": c, "id": c} for c in df.columns],
            data=df.to_dict("records"),
            page_size=15,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"},
        ),
        html.Div(id="links"),
    ]
)


@app.callback(
    Output("table", "data"),
    [Input({"type": "filter-dropdown", "index": col}, "value") for col in df.columns],
)
def update_table(*selected_vals):
    filtered = df.copy()
    for col, val in zip(df.columns, selected_vals):
        if val is not None:
            mask = (filtered[col] == val) | filtered[col].isna()
            filtered = filtered[mask]
    return filtered.to_dict("records")


@app.callback(
    Output("links", "children"),
    Input("table", "active_cell"),
    Input("table", "data"),
)
def show_links(active_cell, table_data):
    if not active_cell:
        return ""
    row = active_cell["row"]
    if row >= len(table_data):
        return ""
    row_data = table_data[row]
    model_id = row_data.get("id")
    if not model_id:
        return ""
    pattern = os.path.join("results", "figures", f"*_{model_id}.html")
    files = glob.glob(pattern)
    links = [
        html.A(
            os.path.basename(f),
            href=f"/figures/{os.path.basename(f)}",
            target="_blank",
            style={"display": "block"},
        )
        for f in files
    ]
    return html.Div(links)


# Serve static figure files from ./results/figures
# Serve static figure files from ./results/figures
@app.server.route("/figures/<path:filename>")
def serve_figure(filename):
    # Resolve the absolute path to the figure directory relative to the script/cwd
    figures_dir = os.path.abspath(os.path.join("results", "figures"))
    return send_from_directory(figures_dir, filename)
    return send_from_directory(os.path.join("results", "figures"), filename)


if __name__ == "__main__":
    app.run(debug=True)
