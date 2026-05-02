import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


def make_graph(n):
    filename = f"sort_run_n{n}.pkl"
    df = pd.read_pickle(filename)
    
    algos = df["Algorithm"].unique()
    inputs = df.columns.tolist()[2:6] #CV, Cardinality, Skew, Sortedness
    outputs = df.columns.tolist()[6:] #Execution Time, comps, assigns

    #Set up figure and subplot axes, make it large so plots don't overlap
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    colors = {'Bubble Sort': 'red', 'Selection Sort': 'blue', 'Insertion Sort': 'green', 'Merge Sort': 'orange', 'Quick Sort': 'purple'}   

    # 3 row by 4 column subplot graph; each row is a different sorting metric, each column is a different
    #distribution parameter
    for i in range(len(outputs)):
        for j in range(len(inputs)):

            axis = axes[i][j]
            #Plot a line for each algorithm by filtering the dataframe according to the algorithm used, then sort by
            #The distribution parameter for that graph

            dist_param = inputs[j]
            sort_metric = outputs[i]
            for algo in algos:
                filtered = df[df["Algorithm"] == algo]
                sort_by_input = filtered.sort_values(by=dist_param)
                
                axis.plot(sort_by_input[dist_param], sort_by_input[sort_metric], label=algo, color=colors[algo])
                axis.tick_params(axis='both', which='major', labelsize=16)
                axis.tick_params(axis='both', which='minor', labelsize=16)
                axis.set_xlabel(dist_param, fontsize=16)
                axis.set_ylabel(sort_metric, fontsize=16)
                axis.set_yscale('log')
                axis.set_title(f"{sort_metric} vs. {dist_param}", fontsize=16)
                

    #Legend
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    return fig


def main():
    DIST_RUN_SIZES = [100, 1000, 5000]
    for size in DIST_RUN_SIZES:
        graph = make_graph(size)
        graph.savefig(f"sort_run_n{size}")

if __name__ == "__main__":
    main()

