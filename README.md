# Collaborative LLM Fine-tuning Experiments

This repository contains two experiments which form the foundation of our proposal for a open, crowdsource supervised fine-tuning method.

## 1. Point in Space Simulation (`point_in_space_sim`)

This experiment views fine-tuning as vector movement in a multi-dimensional embedding space. It examines whether models can be trained to converge toward specific points (emotional targets) in this space.

### Key Components:

- **Single-Model Convergence**: Validates whether a single fine-tuned model can converge toward a defined emotion target
- **Multi-Model Convergence**: Implements a competitive fine-tuning strategy with selection pressure
  - Generates multiple model variants trained on different feedback subsets at each iteration
  - Selects the best-performing model as the base for the next iteration

The results demonstrate that competitive selection in fine-tuning leads to better performance than single-model approaches, suggesting that selection pressure is an effective mechanism for improving model alignment.

## 2. User Grouping, Evaluation, and Contribution Analysis (`user_grouping_evaluation_sim`)

This experiment uses the multi-model approach as our foundation and focuses on how we can group users and evaluate model performance to optimize model convergence and our point system's correlation with Shapley values.

### Key Components:

- **User Grouping Methods**:

  - Random Grouping: Users randomly assigned with equal probability
  - ε-greedy: Assignment increasingly favors users with higher contribution scores
  - Interleaved Grouping: Alternates between high and low-performing users to create balanced teams

- **Model Evaluation Metrics**:

  - L2 Norm (Euclidean Distance)
  - L1 Norm (Manhattan Distance)
  - Dot Product

- **Simulation Setup**:
  - Uses MovieLens dataset where each genre represents a dimension in vector space
  - Defines an expert target as a randomly selected user's preference vector
  - Groups users, updates models via weighted centroids, and selects best model for next iteration
  - Computes contribution scores based on marginal improvements
  - Validates scoring mechanism against estimated Shapley values

The experiment aims to identify optimal grouping strategies and evaluation methods while ensuring that contribution tracking correlates with Shapley values and that model converges toward the expert point.

## Installation

To install the project dependencies, ensure you have Python installed, then run the following command:

```bash
pip install -r requirements.txt
```

## Acknowledgements

- **`SamLowe/roberta-base-go_emotions`**: Used for mapping text to emotional dimensions in the `point_in_space_sim` experiment.
- **SHAP (SHapley Additive exPlanations)**: https://github.com/shap/shap - Utilized for kernel Shapley value computations in the `user_grouping_evaluation_sim` experiment.
- **MovieLens Dataset**: Provided the data for user preference modeling and simulation in the `user_grouping_evaluation_sim` experiment.
