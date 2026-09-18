import os
import random
import msgpack
from datetime import datetime

class SparseEstimator:
    def __init__(self):
        # generate a random unseeded 10-digit numeric string for use as an identifier
        self.id = ''.join(str(random.randint(0, 9)) for _ in range(10))
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.header = {"id": self.id, "date": date}

    def update_header(self, **kwargs):
        """
        Add a new set of keyword arguments to the header.
        """
        self.header.update(kwargs)

    def save_fig(self, fig, name, save_dir="results"):
        """
        Save a figure to disk. Inputs:
        fig: Plotly figure
        name: str
        """
        # save the figure to save_dir / figures / "{name}_{self.id}.html"
        fig_path = os.path.join(save_dir, "figures", f"{name}_{self.id}.html")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.write_html(fig_path)

    def save_header(self, save_dir="results"):
        # save self.header to save_dir / headers / "{self.id}.msgpack"
        header_path = os.path.join(save_dir, "headers", f"{self.id}.msgpack")
        os.makedirs(os.path.dirname(header_path), exist_ok=True)
        with open(header_path, "wb") as f:
            f.write(msgpack.packb(self.header, use_bin_type=True))