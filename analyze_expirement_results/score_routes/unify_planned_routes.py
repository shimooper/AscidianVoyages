from pathlib import Path
import pandas as pd

PLANNED_SUMMER_ROUTES_PATH = Path('planned_routes/sampled_summer_routes.csv')
PLANNED_WINTER_ROUTES_PATH = Path('planned_routes/sampled_winter_routes.csv')
ALL_PLANNED_ROUTES_PATH = Path('planned_routes/all_sampled_routes.csv')


def unify_planned_routes(planned_summer_routes_path: Path, planned_winter_routes_path: Path):
    summer_routes = pd.read_csv(planned_summer_routes_path)
    winter_routes = pd.read_csv(planned_winter_routes_path)
    summer_routes['Season'] = 'summer'
    winter_routes['Season'] = 'winter'

    # Concatenate the two DataFrames
    unified_routes = pd.concat([summer_routes, winter_routes], ignore_index=True)
    unified_routes.to_csv(ALL_PLANNED_ROUTES_PATH, index=False)


if __name__ == "__main__":
    unify_planned_routes(PLANNED_SUMMER_ROUTES_PATH, PLANNED_WINTER_ROUTES_PATH)
