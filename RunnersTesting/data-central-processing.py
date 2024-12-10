import os.path

folder = "/home/zgrate/PycharmProjects/mastersapp/RunnersTesting/shared/20241209_16341733758471"

import pandas as pd
import matplotlib.pyplot as plt


def draw_graphs(folder):
    p = os.path.join(folder, "training_logs.csv")

    df = pd.read_csv(p, sep=";")

    df.info()

    plt.scatter(df['epoch'], df['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Loss Function on Training")
    plt.title("Central Training")

    plt.savefig("training_logs.png")


draw_graphs(folder)
