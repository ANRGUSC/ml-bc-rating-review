# User Grouping and MovieLens Data Analysis

This project provides simulations for analyzing how different combinations of grouping and evaluation methods and changes in `num_users` and `num_groups` impact model convergence and our point system's correlation with Shapley values.

---

## Project Structure

- **`distance_plots/`**: Contains plots demonstrating convergence distance for each grouping method and evaluation method combination
- **`graphing_stats/`**: Contains `.csv` files with raw and average statistics for different simulation runs for different values of `num_users` and `num_groups`
- **`movielens_data/`**: Houses the user preference data used to run this simulation from the MovieLens dataset.
- **`param_sweep/`**: Parameter sweep where we explore the simulation impact across different values of `num_users` and `num_groups`.
- **`pearson_plots/`**: Plots inverse pearson correlation (point values vs. estimated shapley values) and convergence distance. Points near the bottom left are high performing evaluation and grouping method combinations.
- **`subset_influence/`**: Used to visualize how the selected user subset influences model movement toward the expert point.
- **`user_grouping_evaluation.py`**: Main script for running simulations and generating evaluation data.

---

## How to Run the Project

### Run User Grouping Evaluation

Run the `user_grouping_evaluation.py` script to simulate grouping and evaluation methods:

```bash
python user_grouping_evaluation_sim/user_grouping_evaluation.py
```

### Run Parameter Sweep Simulation

Run the `param_sweep_sim.py` script to simulate how changes in `num_users` and `num_groups` impact model convergence and point vs. shapley correlation:

```bash
python param_sweep/param_sweep_sim.py
```

---

### Analyze Parameter Sweep Results

Run the `analyze_param_sweep.py` generate statistics and visualizations for how model convergence and shapley correlation are impacted as we adjust `num_users` and `num_groups`

```bash
python param_sweep/analyze_param_sweep.py
```

---

### Analyze Subset Influence

```bash
python subset_influence/subset_influence.py
```

**This script:**

- Generates a plot showing how a user subset pulls the model toward its centroid.
- Saves the plot as `subset_influence.pdf` in the same directory.

---
