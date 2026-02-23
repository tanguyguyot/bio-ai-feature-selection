from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


def evaluate_combination(args: tuple) -> tuple:
    (
        selected_columns,
        feature_columns,
        dataset,
        y,
        penalty_factor,
        test_size,
        classifier_type,
        max_depth,
    ) = args
    X = dataset[list(selected_columns)]
    col_indexes = tuple([feature_columns.index(col) for col in selected_columns])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=123
    )

    if classifier_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=30,
            max_depth=None,
            min_samples_split=2,
            max_features="sqrt",
            min_impurity_decrease=0.0,
            criterion="gini",
            random_state=456,
            n_jobs=1,
        )
    else:
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=456)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    error = 1 - accuracy
    binary_rep = to_binary_representation(col_indexes, len(feature_columns))
    return binary_rep, {
        "Error": error,
        "Features selected": col_indexes,
        "Lookup value": error + len(selected_columns) * penalty_factor,
        "Penalty factor": penalty_factor,
    }


def get_table(
    dataset: pd.DataFrame,
    feature_columns: list,
    y: pd.Series,
    penalty_factor: float = 0.01,
    test_size: float = 0.3,
    classifier_type: str = "rf",
    max_depth: int = 3,
) -> dict:
    error_table = {}
    tasks = []
    for i in tqdm(range(1, len(feature_columns) + 1)):
        for selected_columns in combinations(feature_columns, i):
            args = (
                selected_columns,
                feature_columns,
                dataset,
                y,
                penalty_factor,
                test_size,
                classifier_type,
                max_depth,
            )
            tasks.append(args)

    with ProcessPoolExecutor() as executor:
        for result in tqdm(
            executor.map(evaluate_combination, tasks), total=len(tasks)
        ):
            bitstring, entry = result
            error_table[bitstring] = entry

    error_table["0" * len(feature_columns)] = {
        "Error": 1,
        "Features selected": tuple(),
        "Lookup value": 1,
        "Penalty factor": penalty_factor,
    }
    return error_table


def change_penalty(table: dict, new_penalty: float) -> dict:
    for key, value in table.items():
        table[key]["Penalty factor"] = new_penalty
        table[key]["Lookup value"] = (
            value["Error"] + len(value["Features selected"]) * new_penalty
        )
    return table


def to_binary_representation(indexes: tuple, length: int) -> str:
    binary_rep = "0" * length
    for i in indexes:
        binary_rep = binary_rep[:i] + "1" + binary_rep[i + 1 :]
    return binary_rep


def export_to_csv(complete_table: dict, dataset_name: str, output_dir: str = "outputs") -> None:
    df = pd.DataFrame.from_dict(
        complete_table,
        orient="index",
        columns=["Error", "Features selected", "Lookup value", "Penalty factor"],
    )
    df.index.name = "Binary representation"
    df.to_csv(f"{output_dir}/{dataset_name}_complete_table.csv", index=True)
    print(f"Complete table saved to {output_dir}/{dataset_name}_complete_table.csv")


def csv_to_dict(csv_file: str, has_headers: bool = True) -> dict:
    if has_headers:
        table = pd.read_csv(csv_file, dtype={"Binary representation": str})
        table.index = table["Binary representation"]
        table = table.drop(columns=["Binary representation"])
    else:
        table = pd.read_csv(csv_file, dtype={0: str}, header=None)
        table.index = table[0]
        table = table.drop(columns=[0])
    return table.to_dict(orient="index")
