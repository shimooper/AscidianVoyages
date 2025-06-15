import pandas as pd
from pathlib import Path

DATA_PATH = Path(r"C:\repos\GoogleShips\analyze_expirement_results\data\preprocess\Final_Data_Voyages_Processed_30.csv")

MIN_ROUTE_LENGTH = 6


def remove_short_routes():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data from {DATA_PATH}.")

    df = df[(df['dying_day'].isna()) | (df['dying_day'] >= MIN_ROUTE_LENGTH)]
    print(f"Removed routes shorter than {MIN_ROUTE_LENGTH} days.")

    df.to_csv(DATA_PATH.parent / f"{DATA_PATH.stem}_filter_short_routes.csv", index=False)
    print("Short routes removed and saved to new file.")


if __name__ == "__main__":
    remove_short_routes()
