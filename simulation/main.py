from math import factorial
from typing import Iterable
import numpy as np
import plotly.express as px
import pandas as pd
from functools import partial
from itertools import combinations
import pathlib

from group_func_helper import permutation_split
from feature_func_helper import gaussian, uniform, triangular
from value_func_helper import dot_product, l2_norm, l2_norm_mean, l2_norm_median, dot_product_mean, dot_product_median
from winning_func_helper import max_points
#Add a global input that controls numpy's random seed

thisdir = pathlib.Path(__file__).parent.resolve()

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
            random_seed: int = 0):

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

    # plot actual Shapley values against points
    df = pd.DataFrame({
        'user': list(range(num_users)),
        'shapley': shapely_values,
        'points': points
    })

    # save stuff
    #TODO: Save data for each round in CSV, maybe add a separate round csv file
    #TODO: Consider storing it in a JSON, target: re-run the simulation from output files
    savedir.mkdir(exist_ok=True, parents=True)
    df.to_csv(savedir / 'sim.csv')
    fig = px.scatter(
        df, x='shapley', y='points',
        hover_name='user',
        template='plotly_white',
    )
    fig.update_traces(marker=dict(size=12))
    fig.write_image(str(savedir / 'shapley.png'))


def main():
    run_sim(thisdir / 'simulation_1', num_users=10, num_experts=2, num_groups=2, num_rounds=10000)
    run_sim(thisdir / 'simulation_2', expert_distribution_method='gaussian_mean_0.5',num_users=10, num_experts=2, num_groups=2, num_rounds=10000)

if __name__ == '__main__':
    main()
