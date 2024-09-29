import os
import pandas as pd
from utils import OUTPUTS_DIR
from sample_routes import sample_winter_subroutes, convert_to_summer_routes

ROUTES_TO_SAMPLE = 1
NEW_SAMPLES_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "new_sampled_routes")

EXISTING_WINTER_ROUTES_PATH = os.path.join(OUTPUTS_DIR, "sampled_winter_routes.csv")


def main():
    routes_with_conditions_path = os.path.join(OUTPUTS_DIR, "routes_with_conditions.csv")
    processed_routes_df = pd.read_csv(routes_with_conditions_path)
    processed_routes_df['Time'] = pd.to_datetime(processed_routes_df['Time'], errors='coerce')

    existing_winter_routes_df = pd.read_csv(EXISTING_WINTER_ROUTES_PATH)
    existing_winter_routes_df['Time'] = pd.to_datetime(existing_winter_routes_df['Time'], errors='coerce', dayfirst=True)
    existing_winter_routes_df['Year'] = existing_winter_routes_df['Time'].dt.year

    # Create a list of unique combinations of category_column and year
    existing_winter_routes_ids = list(existing_winter_routes_df[['Ship', 'Year']].drop_duplicates().itertuples(index=False, name=None))

    os.makedirs(NEW_SAMPLES_OUTPUTS_DIR, exist_ok=True)
    winter_routes = sample_winter_subroutes(processed_routes_df, ROUTES_TO_SAMPLE, NEW_SAMPLES_OUTPUTS_DIR, existing_winter_routes_ids)
    convert_to_summer_routes(winter_routes, NEW_SAMPLES_OUTPUTS_DIR)


if __name__ == "__main__":
    main()
