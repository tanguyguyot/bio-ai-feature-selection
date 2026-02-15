import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm


def is_local_minimum(bitstring: str, table: dict, use_lookup: bool = False) -> bool:
    if use_lookup:
        bitstring_value = table[bitstring]["Lookup value"]
    else:
        bitstring_value = table[bitstring]["Error"]

    neighbors = []
    for i in range(len(bitstring)):
        if bitstring[i] == "1":
            neighbors.append(bitstring[:i] + "0" + bitstring[i + 1 :])
        else:
            neighbors.append(bitstring[:i] + "1" + bitstring[i + 1 :])

    if use_lookup:
        neighbors_values = [table[neighbor]["Lookup value"] for neighbor in neighbors]
    else:
        neighbors_values = [table[neighbor]["Error"] for neighbor in neighbors]

    return min(neighbors_values) >= bitstring_value


def get_local_minimums(table: dict, use_lookup: bool = False) -> list:
    local_minimums = []
    for bitstring in table:
        if bitstring.count("1") == 0:
            continue
        if is_local_minimum(bitstring, table, use_lookup):
            if use_lookup:
                local_minimums.append(
                    (bitstring, int(bitstring, 2), table[bitstring]["Lookup value"])
                )
            else:
                local_minimums.append(
                    (bitstring, int(bitstring, 2), table[bitstring]["Error"])
                )
    local_minimums.sort(key=lambda x: x[2])
    return local_minimums


def visualization_2d(complete_table: dict, dataset_name: str, output_dir: str = "outputs") -> None:
    plt.close()
    X = []
    y = []
    for key, value in tqdm(complete_table.items()):
        if key.count("1") == 0:
            continue
        X.append(int(key, 2))
        y.append(value["Lookup value"])
    X = np.array(X)
    y = np.array(y)
    ind = np.argsort(X)

    local_minimums = get_local_minimums(complete_table, use_lookup=True)
    X_locals = [x[1] for x in local_minimums]
    y_locals = [x[2] for x in local_minimums]

    plt.plot(X[ind], y[ind], color="gray", linewidth=0.5, alpha=0.8)
    plt.scatter(X, y, alpha=0.5, color="green", s=10)
    plt.scatter(
        X_locals,
        y_locals,
        label=f"Local optimums ({len(X_locals)})",
        facecolors="none",
        edgecolors="blue",
    )
    plt.scatter(
        X_locals[0],
        y_locals[0],
        label=f"Global minimum: {y_locals[0]:.4f}",
        facecolors="none",
        edgecolors="red",
    )

    plt.title(f"2D visualization of {dataset_name}")
    plt.xlabel("Binary representation (as int)")
    plt.ylabel("Lookup value")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_dir}/plots/{dataset_name}_plot.png")
    plt.show()
    plt.close()


def hinged_bitstring_map(
    complete_table: dict, dataset_name: str, output_dir: str = "outputs"
) -> None:
    plt.close()

    sample_key = next(iter(complete_table.keys()))
    length = len(sample_key)
    half_length = length // 2
    X = []
    y = []
    lookup_values = []
    for key, value in tqdm(complete_table.items()):
        X.append(int(key[:half_length], 2))
        y.append(int(key[half_length:], 2))
        lookup_values.append(value["Lookup value"])
    X = np.array(X)
    y = np.array(y)
    lookup_values = np.array(lookup_values)

    def color_value(x: float) -> np.ndarray:
        start_color = np.array([0, 100, 0]) / 255
        mid_color = np.array([255, 245, 245]) / 255
        end_color = np.array([128, 0, 80]) / 255

        if x <= 0.5:
            t = x / 0.5
            color = (1 - t) * start_color + t * mid_color
        else:
            t = (x - 0.5) / 0.5
            color = (1 - t) * mid_color + t * end_color
        return color

    color_gradient = [color_value(x) for x in lookup_values]

    n_colors = 256
    cmap = ListedColormap([color_value(x) for x in np.linspace(0, 1, n_colors)])
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(length - 2, length))

    sc = ax.scatter(X, y, c=color_gradient)

    local_minimums = get_local_minimums(complete_table, use_lookup=True)
    X_locals = [int(x[0][:half_length], 2) for x in local_minimums]
    y_locals = [int(x[0][half_length:], 2) for x in local_minimums]
    X_locals = np.array(X_locals)
    y_locals = np.array(y_locals)

    global_minimum = local_minimums[0]
    X_locals = np.append(X_locals, int(global_minimum[0][:half_length], 2))
    y_locals = np.append(y_locals, int(global_minimum[0][half_length:], 2))

    ax.scatter(
        X_locals,
        y_locals,
        label=f"Local optimums ({len(X_locals)})",
        facecolors="none",
        edgecolors="blue",
    )
    ax.scatter(
        X_locals[-1],
        y_locals[-1],
        label=f"Global minimum: {global_minimum[2]:.4f}",
        s=50,
        facecolors="none",
        edgecolors="red",
    )

    plt.title(f"HBM of {dataset_name}")
    plt.xlabel("Binary representation part 1")
    plt.ylabel("Binary representation part 2")
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.1, aspect=40, shrink=0.8)
    cbar.set_label("Lookup value", fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_xticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/{dataset_name}_hbm.png")
    plt.show()
    plt.close()
