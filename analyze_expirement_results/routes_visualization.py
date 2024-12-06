import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_timeline(routes_path, condition_label, condition_full_name, y_axis_label, output_dir):
    fig, ax = plt.subplots(figsize=(15, 10))

    df = pd.read_csv(routes_path)
    lived_columns = sorted([col for col in df.columns if 'Lived' in col], key=lambda x: int(x.split()[1]))
    condition_columns = sorted([col for col in df.columns if condition_label in col], key=lambda x: int(x.split()[1]))

    for index, row in df.iterrows():
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

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_dir / f'routes_timeline_{condition_full_name.lower()}', dpi=300)


def plot_timelines(routes_path, output_dir):
    plot_timeline(routes_path, 'Temp', 'Temperature', 'Temperature (celsius)', output_dir / 'routes_visualizations')
    plot_timeline(routes_path, 'Salinity', 'Salinity', 'Salinity (ppt)', output_dir / 'routes_visualizations')
