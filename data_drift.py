import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import ks_2samp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataDriftMonitor(BaseEstimator, TransformerMixin):
    def __init__(self, train_data, buckets=10, show_plots=False):
        self.train_data = train_data
        self.buckets = buckets
        self.show_plots = show_plots

    def ks_test(self, train_data, new_data):
        ks_results = {}
        for column in train_data.columns:
            if train_data[column].dtype in [np.float64, np.int64]:
                ks_stat, p_value = ks_2samp(train_data[column], new_data[column])
                ks_results[column] = p_value
        return ks_results

    def calculate_psi(self, expected, actual, buckets=10):
        def scale_range(input, min_val, max_val):
            input = np.clip(input, min_val, max_val)
            input -= min(input)
            input /= (max(input) - min(input))
            return input

        # Calculate bin edges
        breakpoints = np.linspace(0, 1, buckets + 1)
        expected_bins = np.histogram(scale_range(expected, 0, 1), bins=breakpoints)[0]
        actual_bins = np.histogram(scale_range(actual, 0, 1), bins=breakpoints)[0]

        # Convert to percentages
        expected_percents = expected_bins / len(expected)
        actual_percents = actual_bins / len(actual)

        # Handle zero percentages
        expected_percents = np.clip(expected_percents, 1e-10, 1)
        actual_percents = np.clip(actual_percents, 1e-10, 1)

        # Calculate PSI
        psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
        return psi_value

    def calculate_psi_categorical(self, expected, actual):
        # Calculate category distributions
        expected_dist = expected.value_counts(normalize=True)
        actual_dist = actual.value_counts(normalize=True)
        
        # Combine distributions and handle missing categories
        all_categories = set(expected_dist.index).union(set(actual_dist.index))
        expected_dist = expected_dist.reindex(all_categories, fill_value=0)
        actual_dist = actual_dist.reindex(all_categories, fill_value=0)
        
        # Handle zero percentages
        expected_dist = np.clip(expected_dist, 1e-10, 1)
        actual_dist = np.clip(actual_dist, 1e-10, 1)
        
        # Calculate PSI
        psi_value = np.sum((expected_dist - actual_dist) * np.log(expected_dist / actual_dist))
        return psi_value

    def plot_results(self, ks_results, psi_numeric_results, psi_categorical_results):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        # Plotting PSI for Numerical Features
        sns.barplot(ax=axes[0], x=list(psi_numeric_results.keys()), y=list(psi_numeric_results.values()))
        axes[0].set_title('PSI for Numerical Features')
        axes[0].set_ylabel('PSI Value')
        axes[0].tick_params(axis='x', rotation=45)

        # Plotting PSI for Categorical Features
        sns.barplot(ax=axes[1], x=list(psi_categorical_results.keys()), y=list(psi_categorical_results.values()))
        axes[1].set_title('PSI for Categorical Features')
        axes[1].set_ylabel('PSI Value')
        axes[1].tick_params(axis='x', rotation=45)

        # Plotting KS Test Results
        sns.barplot(ax=axes[2], x=list(ks_results.keys()), y=list(ks_results.values()))
        axes[2].set_title('KS Test Results')
        axes[2].set_ylabel('p-value')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        
        # Perform data drift monitoring
        ks_results = self.ks_test(self.train_data, X)

        # Calculate PSI results for numeric features
        psi_numeric_results = {
            column: self.calculate_psi(self.train_data[column], X[column])
            for column in self.train_data.columns
            if self.train_data[column].dtype in [np.float64, np.int64]
        }

        # Calculate PSI results for categorical features
        psi_categorical_results = {
            column: self.calculate_psi_categorical(self.train_data[column], X[column])
            for column in ['job', 'location', 'marital_status', 'gender']
            if self.train_data[column].dtype == 'object'
        }

        
        if self.show_plots == True:
        # Plot Results
            if len(self.train_data)!= len(X):
                # Print Results
                print("KS Test Results:", ks_results)
                print("PSI Results (Numeric Features):", psi_numeric_results)
                print("PSI Results (Categorical Features):", psi_categorical_results)

                self.plot_results(ks_results, psi_numeric_results, psi_categorical_results)

        return X
