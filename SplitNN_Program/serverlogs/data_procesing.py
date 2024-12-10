import json
import os

import pandas as pd
import matplotlib.pyplot as plt


def get_details_files(file="details.json"):
    for f in os.listdir():
        if os.path.isdir(f):
            details_file = os.path.join(f, file)
            if os.path.exists(details_file):
                yield os.path.abspath(details_file)


def get_folders():
    for f in os.listdir():
        if os.path.isdir(f):
            yield f


def sum_aggregate_transfer(key: str = "clients", data_transfer_field="data_transfer_len", xlabel=None, ylabel=None,
                           title=None):
    x = []
    y = []
    for f in get_folders():
        with open(os.path.join(f, "details.json")) as fil:
            details = json.load(fil)
        p = pd.read_csv(os.path.join(f, "data_transfer_log.csv"))

        x.append(details[key])
        y.append(p[data_transfer_field].sum())

    plt.scatter(x, y)
    plt.xlabel(key if xlabel is None else xlabel)
    plt.ylabel(data_transfer_field if ylabel is None else ylabel)
    plt.title(f"{key}_to_{data_transfer_field}" if title is None else title)
    plt.savefig(f"{key}_to_{data_transfer_field}.png")
    plt.close()


def compare_field(key: str = "clients", field: str = "server_epoch", in_results=True):
    x = []
    y = []
    for f in get_details_files():
        with open(f, "r") as json_file:
            j = json.load(json_file)
            x.append(j[key])
            y.append(j['results'][field] if in_results else j[field])

    plt.scatter(x, y)
    plt.xlabel(key)
    plt.ylabel(field)
    plt.text("TEST")
    plt.savefig(f"{key}_to_{field}.png")
    plt.close()


sum_aggregate_transfer()
exit(0)

compare_field()
compare_field(field="average_total_time")
compare_field(field="average_comm_time")
compare_field(field="max_client_epoch")
