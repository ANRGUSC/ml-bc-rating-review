from math import factorial
from typing import Iterable
import numpy as np
import plotly.express as px
import pandas as pd
from functools import partial
from itertools import combinations

from feature_gen import FeatureGenerator
from value_func import ValueFunction
from feature_gen import Feature_Methods
from group_func import GroupSplitting

def all_subsets(elements: Iterable, exclude: Iterable = []) -> Iterable:
    """Returns all subsets (of length > 0) of elements excluding those in exclude"""
    # yield empty set
    yield []
    for i in range(1, len(elements) + 1):
        for subset in combinations(elements, i):
            if not any(x in subset for x in exclude):
                yield subset

def main():
    # users are represented by a vector of features
    num_features = 3
    num_experts = 2
    num_users = 10

    # users are represented by a vector of features
    # users = FeatureGenerator.uniform(num_users, num_features)
    # users = Feature_Methods.get_feature_generator(Feature_Methods.UNIFORM)(num_users, num_features)
    users = Feature_Methods.get_feature_generator(Feature_Methods.GAUSSIAN)(num_users, num_features, mean=0.5, std_dev=0.1)

    # experts are represented by a vector of features
    experts = FeatureGenerator.uniform(num_experts, num_features)

    # aggregate function is average of user features
    aggregate = partial(np.mean, axis=0)

    # value function is max dot product of aggregate features and expert features
    value = lambda coalition: 0 if len(coalition) <= 0 else np.max(np.dot(experts, aggregate(coalition)))    

    # Compute actual Shapley values
    all_users = list(range(num_users))
    shapely_values = np.zeros(num_users)
    for user in range(num_users):
        # iterate over all subsets without user
        for subset in all_subsets(all_users, exclude=[user]):
            # compute value of subset
            subset_value = value(users[list(subset)])
            # subset_value = ValueFunction.dot_product(users[list(subset)])
            # compute value of subset with user
            subset_with_user_value = value(users[list(subset) + [user]])
            # compute marginal contribution of user
            marginal_contribution = subset_with_user_value - subset_value
            # update user value
            weight = factorial(len(subset)) * factorial(num_users - len(subset) - 1) / factorial(num_users)
            shapely_values[user] += weight * marginal_contribution


    # simulate protocol which estimates Shapley values by iteratively splitting users into two groups and asking experts to rank them
    # experts are asked to rank users in each group
    num_rounds = 10000
    num_groups = 2
    points = np.zeros(num_users)
    for i in range(num_rounds):
        # random permutation of users
        # permutation = np.random.permutation(num_users)
        # split users into groups
        groups = GroupSplitting.permutation_split(num_users, num_groups)
        # ask experts to rank users in each group (apply value function to each group)
        group_values = np.array([value(users[group]) for group in groups])
        # give a point to each user in the winning group
        winning_group = np.argmax(group_values)
        points[groups[winning_group]] += 1

    # plot actual Shapley values against points
    df = pd.DataFrame({
        'user': list(range(num_users)),
        'shapley': shapely_values,
        'points': points
    })

    fig = px.scatter(
        df, x='shapley', y='points',
        hover_name='user',
        template='plotly_white',
    )
    fig.update_traces(marker=dict(size=12))
    fig.write_image('shapley.png')


if __name__ == '__main__':
    main()
