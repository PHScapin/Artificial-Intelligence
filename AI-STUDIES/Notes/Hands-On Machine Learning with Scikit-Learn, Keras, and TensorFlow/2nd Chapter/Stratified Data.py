"""
Context: This script prepares a housing dataset for stratified analysis by creating an "income_cat" column 
based on "median_income" to categorize income levels. It then splits the data into training and 
test sets using both stratified and random sampling, preserving income category proportions to 
avoid bias in machine learning models. Finally, it compares the category proportions to validate 
the representativeness of the sampled sets.
"""

# Ensure Python and Scikit-Learn versions
import sys
assert sys.version_info >= (3, 7)

from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

# Libraries
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import tarfile
import urllib.request
import numpy as np

def load_housing_data():
    """
    Load the housing data from a tarball file. If the file does not exist, it will be downloaded and extracted.

    Returns:
        DataFrame: The housing data in a pandas DataFrame.
    """ 
    datasets_path = Path(__file__).resolve().parents[3] / "Datasets"
    tarball_path = datasets_path / "housing.tgz"

    if not tarball_path.is_file():
        datasets_path.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=datasets_path)  # Extract within the Datasets folder

    return pd.read_csv(datasets_path / "housing/housing.csv")

def add_income_category(data):
    """
    Adds an 'income_cat' column to the dataset by binning the 'median_income' column.

    Parameters:
    - data: DataFrame
        The dataset containing the "median_income" column.
    
    Returns:
    - DataFrame: The dataset with an added "income_cat" column.
    """
    data["income_cat"] = pd.cut(data["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    return data

def income_cat_proportions(data):
    """
    Calculates the proportion of each income category in the dataset.

    Parameters:
    - data: DataFrame
        The dataset containing the "income_cat" column.

    Returns:
    - Series: Proportions of income categories.
    """
    return data["income_cat"].value_counts(normalize=True)

def compare_income_category_proportions(housing_data, strat_test_set, test_set):
    """
    Compares the income category proportions across the full dataset, the stratified test set, 
    and the random test set.

    Args:
        housing_data (DataFrame): The full dataset.
        strat_test_set (DataFrame): The stratified test set.
        test_set (DataFrame): The randomly split test set.

    Returns:
        DataFrame: A comparison table with income category proportions for each dataset split.
    """
    compare_props = pd.DataFrame({
        "Overall %": income_cat_proportions(housing_data),
        "Stratified %": income_cat_proportions(strat_test_set),
        "Random %": income_cat_proportions(test_set),
    }).sort_index()
    compare_props.index.name = "Income Category"
    compare_props["Strat. Error %"] = (compare_props["Stratified %"] / compare_props["Overall %"] - 1) * 100
    compare_props["Rand. Error %"] = (compare_props["Random %"] / compare_props["Overall %"] - 1) * 100

    return compare_props

def shuffle_and_split_data(dataset, test_ratio):
    """
    Shuffle the dataset and split it into training and test sets based on the provided test_ratio. 
    This will randomly create two subdivisions: training set and test set.

    Parameters:
    - dataset: pandas DataFrame
        The dataset to be shuffled and split into training and test sets.
    - test_ratio: float
        The proportion of the dataset to be used for the test set (e.g., 0.2 for 20%).

    Returns:
    - train_set: pandas DataFrame
        The training set, which contains (1 - test_ratio) of the original dataset.
    - test_set: pandas DataFrame
        The test set, which contains test_ratio of the original dataset.
    """
    shuffled_indices = np.random.permutation(len(dataset))
    test_set_size = int(len(dataset) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return dataset.iloc[train_indices], dataset.iloc[test_indices]

def stratified_shuffle_and_split_data(dataset, data_splitter):
    """
    Generates stratified splits of a dataset based on a categorical column and visualizes income distribution.

    This function uses a pre-configured `StratifiedShuffleSplit` object to split the dataset in a stratified
    way based on the "income_cat" column of the `dataset`, which is created by binning the "median_income" column.
    The stratified splitting ensures that the proportion of classes in the "income_cat" column is maintained
    in both the training and testing sets.

    Args:
        dataset (DataFrame): The input dataframe containing the data to be split.
        data_splitter (StratifiedShuffleSplit): A `StratifiedShuffleSplit` object configured externally.

    Returns:
        list: A list containing pairs of dataframes. Each pair consists of a training and testing set.
    """
    stratified_splits = []
    for train_index, test_index in data_splitter.split(dataset, dataset["income_cat"]):
        stratified_train_set_n = dataset.iloc[train_index]
        stratified_test_set_n = dataset.iloc[test_index]
        stratified_splits.append([stratified_train_set_n, stratified_test_set_n])

    return stratified_splits


# Pipeline

## Load housing data
housing_data = load_housing_data()

## Creates a random seed to padronize the answers
np.random.seed(42) # 42 is just my lucky number, don't worry about what that means

## Add income category to housing data
housing_data = add_income_category(housing_data)

## Randomizing and splitting data
train_set, test_set = shuffle_and_split_data(housing_data, 0.2)

## Stratified randomization and splitting data
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
stratified_train_set, stratified_test_set = stratified_shuffle_and_split_data(housing_data, splitter)[0]

## Compare proportions
compare_props = compare_income_category_proportions(housing_data, stratified_test_set, test_set)

## Print the results
print(compare_props.to_string()) 