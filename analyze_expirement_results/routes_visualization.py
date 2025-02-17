import os
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_column_groups_sorted


def plot_timeline(routes_df, lived_columns, condition_columns, condition_full_name, y_axis_label, output_dir):
    fig, ax = plt.subplots(figsize=(15, 10))

    for index, row in routes_df.iterrows():
        survival_values = row[lived_columns].values
        condition_values = row[condition_columns].values
        days = range(len(survival_values))

        ax.plot(days, condition_values, color='gray', alpha=0.3)  # Plot temperature line

        alive_days = [day for day, status in zip(days, survival_values) if status == 0]
        dead_days = [day for day, status in zip(days, survival_values) if status == 1]

        alive_conditions = [condition for condition, status in zip(condition_values, survival_values) if status == 0]
        dead_conditions = [condition for condition, status in zip(condition_values, survival_values) if status == 1]

        ax.scatter(alive_days, alive_conditions, color='green', alpha=0.5, label='Alive' if index == 0 else "")
        ax.scatter(dead_days, dead_conditions, color='red', alpha=0.5, label='Dead' if index == 0 else "")

    ax.set_title(f'Timeline of the Effect of {condition_full_name} on Ascidians Survival')
    ax.set_xlabel('Day')
    ax.set_ylabel(y_axis_label)
    ax.legend(loc='center right')
    ax.grid(True)

    fig.savefig(output_dir / f'routes_timeline_{condition_full_name.lower()}', dpi=300)
    plt.close()


def plot_timelines(config):
    routes_path = config.data_dir_path / 'full.csv'
    df = pd.read_csv(routes_path)
    lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)

    visualizations_dir = config.outputs_dir_path / 'routes_visualizations'
    os.makedirs(visualizations_dir, exist_ok=True)

    plot_timeline(df, lived_columns, temperature_columns, 'Temperature', 'Temperature (celsius)', visualizations_dir)
    plot_timeline(df, lived_columns, salinity_columns, 'Salinity', 'Salinity (ppt)', visualizations_dir)

    routes_summer_df = df[df['Season'] == 'summer']
    plot_timeline(routes_summer_df, lived_columns, temperature_columns, 'Temperature in Summer',
                  'Temperature (celsius)', visualizations_dir)
    plot_timeline(routes_summer_df, lived_columns, salinity_columns, 'Salinity in Summer', 'Salinity (ppt)',
                  visualizations_dir)

    routes_winter_df = df[df['Season'] == 'winter']
    plot_timeline(routes_winter_df, lived_columns, temperature_columns, 'Temperature in Winter',
                  'Temperature (celsius)', visualizations_dir)
    plot_timeline(routes_winter_df, lived_columns, salinity_columns, 'Salinity in Winter', 'Salinity (ppt)',
                  visualizations_dir)
