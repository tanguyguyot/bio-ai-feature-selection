import pandas as pd
from src.table.creation import get_table, export_to_csv
import os

for dataset_path in os.listdir("datasets"):
    if dataset_path.endswith(".csv"):
        print(f"Processing {dataset_path}...")
        dataset = pd.read_csv(f"datasets/{dataset_path}", )
        feature_columns = dataset.columns.drop('Class').tolist()  # All except target
        y = dataset['Class']

        # Create lookup table
        table = get_table(dataset, feature_columns, y, penalty_factor=0.01)

        # Save to CSV
        export_to_csv(table, dataset_path.replace(".csv", ""))