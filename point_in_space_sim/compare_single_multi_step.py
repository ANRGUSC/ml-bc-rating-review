import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_json_data(file_path: Path) -> List[Dict]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_center(neighborhood: List[Dict]) -> np.ndarray:
    """
    Calculate the center of a neighborhood of points.
    
    Args:
        neighborhood: List of dictionaries containing emotion vectors
        
    Returns:
        Center point as numpy array
    """
    # Skip the first element (usually a reference point) and compute mean of the rest
    return np.mean([sentence["emotion"] for sentence in neighborhood[1:]], axis=0)


def calculate_distances(sentences: List[Dict], center: np.ndarray, 
                       k_fraction: Optional[int] = None) -> List[float]:
    """
    Calculate distances from sentences to the center point.
    
    Args:
        sentences: List of dictionaries containing emotion vectors
        center: Center point to calculate distances from
        k_fraction: Optional filter for k_fraction value
        
    Returns:
        List of distances
    """
    if k_fraction is not None:
        filtered_sentences = [s for s in sentences if s.get("k_fraction") == k_fraction]
    else:
        filtered_sentences = sentences
    
    return [
        np.linalg.norm(center - np.array(sentence["emotion"]))
        for sentence in filtered_sentences
    ]


def process_emotion(emotion: str, single_step_dir: Path, multi_step_dir: Path) -> Dict:
    """
    Process data for a single emotion.
    
    Args:
        emotion: Name of the emotion
        single_step_dir: Directory containing single-step data
        multi_step_dir: Directory containing multi-step data
        
    Returns:
        Dictionary containing emotion name and distance metrics
    """
    single_emotion_dir = single_step_dir / emotion
    multi_emotion_dir = multi_step_dir / emotion
    
    # Load neighborhood data (same for both approaches)
    neighborhood = load_json_data(single_emotion_dir / "neighborhood.json")
    center = calculate_center(neighborhood)
    
    # Process Single-Step: Extract distance for k_fraction == 100
    single_sentences = load_json_data(single_emotion_dir / "all_sentences.json")
    single_distances = calculate_distances(single_sentences, center, k_fraction=100)
    single_distance = np.mean(single_distances) if single_distances else np.nan
    
    # Process Multi-Step: Extract distance from all_sentences_3.json
    multi_sentences = load_json_data(multi_emotion_dir / "all_sentences_3.json")
    multi_distances = calculate_distances(multi_sentences, center)
    multi_distance = np.mean(multi_distances) if multi_distances else np.nan
    
    return {
        "Emotion": emotion,
        "Single-Step Distance": single_distance,
        "Multi-Step Distance": multi_distance
    }


def calculate_improvement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate improvement percentage between single-step and multi-step approaches.
    
    Args:
        df: DataFrame with Single-Step and Multi-Step distances
        
    Returns:
        DataFrame with added Improvement column
    """
    df = df.copy()
    
    # Calculate improvement percentage
    df["Improvement (%)"] = (
        (df["Single-Step Distance"] - df["Multi-Step Distance"]) / df["Single-Step Distance"]
    ) * 100
    
    # Handle division by zero and other special cases
    df["Improvement (%)"] = df["Improvement (%)"].replace([np.inf, -np.inf], np.nan)
    
    return df


def create_comparison_plot(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create and save a grouped bar chart comparing single-step and multi-step distances.
    
    Args:
        df: DataFrame with distance data
        output_path: Path to save the plot
    """
    plt.figure(figsize=(14, 6))
    colors = sns.color_palette("Set1", 2)
    
    # Create grouped bar chart
    sns.barplot(x="Emotion", y="Single-Step Distance", data=df, color=colors[0], label="Single-Step")
    sns.barplot(x="Emotion", y="Multi-Step Distance", data=df, color=colors[1], label="Multi-Step")
    
    # Add formatting
    plt.xticks(rotation=90)
    plt.ylabel("Distance to Target Neighborhood")
    plt.xlabel("Emotion")
    plt.title("Single-Step vs Multi-Step Distance Comparison in the Final Iteration")
    plt.legend()
    plt.grid(axis="y")
    
    # Save the plot with tight borders
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    

def main():
    """Main execution function."""
    # Define directories
    script_dir = Path(__file__).parent.absolute()
    single_step_dir = script_dir / "single_step/output"
    multi_step_dir = script_dir / "multi_step/output"
    
    # Create output paths
    results_csv_path = script_dir / "comparison_results.csv"
    plot_path = script_dir / "single_vs_multi_distance.pdf"
    
    # Get list of emotion directories
    emotions = [folder for folder in os.listdir(single_step_dir) 
                if os.path.isdir(single_step_dir / folder)]
    
    # Process each emotion
    results = []
    for emotion in emotions:
        emotion_data = process_emotion(emotion, single_step_dir, multi_step_dir)
        results.append(emotion_data)
    
    # Create and process results DataFrame
    df_results = pd.DataFrame(results)
    df_results = calculate_improvement(df_results)
    
    # Save results to CSV
    df_results.to_csv(results_csv_path, index=False)
    
    # Create and save plot
    create_comparison_plot(df_results, plot_path)


if __name__ == "__main__":
    main()