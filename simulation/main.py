from math import factorial
from typing import Iterable
import numpy as np
import plotly.express as px
import pandas as pd
from functools import partial
from itertools import combinations
import pathlib
import json
import inspect

from helpers.group_func_helper import permutation_split
from helpers.feature_func_helper import gaussian, uniform, triangular
from helpers.value_func_helper import dot_product, l2_norm, l2_norm_mean, l2_norm_median, dot_product_mean, dot_product_median
from helpers.winning_func_helper import max_points
#Add a global input that controls numpy's random seed

thisdir = pathlib.Path(__file__).parent.resolve()
sim_results = thisdir / 'results'

feature_methods = {
    'uniform': partial(uniform),
    'gaussian_mean_0.5': partial(gaussian, mean=0.5, std_dev=0.1),
    'gaussian_mean_2': partial(gaussian, mean=2, std_dev=0.1),
    'triangular': partial(triangular, left=0, mode=0.5, right=1),  
}

aggregate_methods = {
    'average': partial(np.mean, axis=0),
    'median': partial(np.median, axis=0)
}

## NOTE some of these may not be necessary to have in a partial since they are not provided different arguments
group_methods = {
    'permutation': partial(permutation_split),
}

value_methods = {
    'dot_product': partial(dot_product),
    'l2_norm': partial(l2_norm),
    'l2_norm_mean': partial(l2_norm_mean),
    'l2_norm_median': partial(l2_norm_median),
    'dot_product_mean': partial(dot_product_mean),
    'dot_product_median': partial(dot_product_median)
}

winning_methods = {
    'max_points': partial(max_points),
    # 'max_points_with_noise': partial(max_points, noise=np.gaus)
}

def save_sim_config(savedir, **kwargs):
    config = {
        'expert_distribution_method': kwargs.get('expert_distribution_method', 'uniform'),
        'user_distribution_method': kwargs.get('user_distribution_method', 'uniform'),
        'aggregate_method': kwargs.get('aggregate_method', 'average'),
        'value_method': kwargs.get('value_method', 'dot_product'),
        'group_method': kwargs.get('group_method', 'permutation'),
        'winning_method': kwargs.get('winning_method', 'max_points'),
        'num_features': kwargs.get('num_features', 3),
        'num_experts': kwargs.get('num_experts', 2),
        'num_users': kwargs.get('num_users', 10),
        'num_groups': kwargs.get('num_groups', 2),
        'num_rounds': kwargs.get('num_rounds', 10000),
        'random_seed': kwargs.get('random_seed', 0),
    }

    with open(savedir / 'config.json', 'w') as f:
        json.dump(config, f)

## can be called to run the simulation from a previous run
def run_sim_from_config(filepath):
    with open(filepath, 'r') as f:
        config = json.load(f)

    run_sim(**config, save_config=False)
        

def all_subsets(elements: Iterable, exclude: Iterable = []) -> Iterable:
    """Returns all subsets (of length > 0) of elements excluding those in exclude"""
    # yield empty set
    for i in range(1, len(elements) + 1):
        for subset in combinations(elements, i):
            if not any(x in subset for x in exclude):
                yield subset


def run_sim(savedir: pathlib.Path,
            expert_distribution_method='uniform',
            user_distribution_method='uniform',
            aggregate_method='average',
            value_method='dot_product',
            group_method='permutation',
            winning_method='max_points',
            num_features: int = 3,
            num_experts: int = 2,
            num_users: int = 10,
            num_groups: int = 2,
            num_rounds: int = 10000,
            random_seed: int = 0,
            save_config: bool = True):

    np.random.seed(random_seed)

    ### Functions to be used in the simulation ###
    # distributions of user and expert features
    config = {
        'expert_distribution': feature_methods[expert_distribution_method],
        'user_distribution': feature_methods[user_distribution_method],
        'aggregate': aggregate_methods[aggregate_method],
        'value': value_methods[value_method],
        'group': group_methods[group_method],
        'winning': winning_methods[winning_method]
    }

    ## This sections grabs the necessary config functions
    users = config['user_distribution'](num_users, num_features)
    #users = Feature_Methods.get_feature_generator(Feature_Methods.GAUSSIAN)(num_users, num_features, mean=0.5, std_dev=0.1)

    # experts are represented by a vector of features
    experts = config['expert_distribution'](num_experts, num_features)

    # aggregate function is average of user features
    # aggregate = partial(np.mean, axis=0)
    aggregate = config['aggregate']

    # value function is max dot product of aggregate features and expert features
    # value = lambda coalition: 0 if len(coalition) <= 0 else np.max(np.dot(experts, aggregate(coalition)))    
    value_func = config['value']
   
    # Compute actual Shapley values
    all_users = list(range(num_users))
    shapely_values = np.zeros(num_users)

    for user in range(num_users):
        # iterate over all subsets without user
        for subset in all_subsets(all_users, exclude=[user]):
            # compute value of subset
            subset_value = value_func(experts, users[list(subset)], aggregate)
            # subset_value = ValueFunction.dot_product(users[list(subset)])
            # compute value of subset with user
            subset_with_user_value = value_func(experts, users[list(subset) + [user]], aggregate)
            # compute marginal contribution of user
            marginal_contribution = subset_with_user_value - subset_value
            # update user value
            weight = factorial(len(subset)) * factorial(num_users - len(subset) - 1) / factorial(num_users)
            shapely_values[user] += weight * marginal_contribution


    # simulate protocol which estimates Shapley values by iteratively splitting users into two groups and asking experts to rank them
    # experts are asked to rank users in each group
    points = np.zeros(num_users)

    round_data = pd.DataFrame()

    for i in range(num_rounds):
        # random permutation of users
        # permutation = np.random.permutation(num_users)
        # split users into groups
        groups = config['group'](num_users, num_groups)

        # ask experts to rank users in each group (apply value function to each group)
        group_values = np.array([value_func(experts, users[group], aggregate) for group in groups])
        # give a point to each user in the winning group
        winning_groups = config['winning'](group_values)
        for winning_group, points_won in winning_groups.items():
            points[groups[winning_group]] += points_won

    #     current_rount = pd.DataFrame({
    #         'user': list(range(num_users)), 
    #         'points_round_{}'.format(i+1): points
    #     })

    #     if round_data.empty:
    #         round_data = current_rount
    #     else:
    #         round_data = round_data.merge(current_rount, on='user')

    savedir.mkdir(exist_ok=True, parents=True)

    # round_data.to_csv(savedir / 'round_data.csv', index=False)

    # plot actual Shapley values against points
    df = pd.DataFrame({
        'user': list(range(num_users)),
        'shapley': shapely_values,
        'points': points
    })

    # save stuff
    #TODO: Fix the round.csv file
    df.to_csv(savedir / 'sim.csv')
    fig = px.scatter(
        df, x='shapley', y='points',
        hover_name='user',
        template='plotly_white',
    )
    fig.update_traces(marker=dict(size=12))
    fig.write_image(str(savedir / 'shapley.png'))

    if save_config:
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)

        config_to_save = {arg: values[arg] for arg in args if arg not in ["frame", "rand_gen_state"]}
        save_sim_config(**config_to_save)

#TODO: Add the presence of the model and initialize it to maybe 60-70% of the expert value.
def main():
    # Default with different numbers
    run_sim(sim_results / 'uniform_avg_dot_max_15u_3e_3g', 
            num_users=15, num_experts=3, num_groups=3, num_rounds=10000)

    # Gaussian distribution for experts
    run_sim(sim_results / 'gauss0.5_avg_dot_max', 
            expert_distribution_method='gaussian_mean_0.5', num_users=10, num_experts=2, num_groups=2, num_rounds=10000)

    # Different Gaussian and median aggregate
    run_sim(sim_results / 'gauss2_median_dot_max', 
            expert_distribution_method='gaussian_mean_2', aggregate_method='median', num_users=10, num_experts=2, num_groups=2, num_rounds=10000)

    # Triangular distribution with L2 norm
    run_sim(sim_results / 'triangular_avg_l2_max', 
            expert_distribution_method='triangular', value_method='l2_norm', num_users=10, num_experts=2, num_groups=2, num_rounds=10000)

    # Uniform distribution with dot_product_mean
    run_sim(sim_results / 'uniform_avg_dotmean_max', 
            value_method='dot_product_mean', num_users=10, num_experts=2, num_groups=2, num_rounds=10000)

    # Gaussian, median aggregate, dot_product_median
    run_sim(sim_results / 'gauss0.5_med_dotmed_max', 
            expert_distribution_method='gaussian_mean_0.5', aggregate_method='median', value_method='dot_product_median', 
            num_users=10, num_experts=2, num_groups=2, num_rounds=10000)

    # Triangular with L2 norm mean
    run_sim(sim_results / 'triangular_avg_l2mean_max', 
            expert_distribution_method='triangular', value_method='l2_norm_mean', num_users=10, num_experts=2, num_groups=2, num_rounds=10000)

    # Gaussian with L2 norm median
    run_sim(sim_results / 'gauss2_avg_l2med_max', 
            expert_distribution_method='gaussian_mean_2', value_method='l2_norm_median', num_users=10, num_experts=2, num_groups=2, num_rounds=10000)

    # Default with more rounds
    run_sim(sim_results / 'uniform_avg_dot_max_20krnds', 
            num_users=10, num_experts=2, num_groups=2, num_rounds=20000)

    # Gaussian mean 0.5 with L2 norm and more rounds
    run_sim(sim_results / 'gauss0.5_avg_l2_max_15krnds', 
            expert_distribution_method='gaussian_mean_0.5', value_method='l2_norm', num_users=10, num_experts=2, num_groups=2, num_rounds=15000)


if __name__ == '__main__':
    main()
