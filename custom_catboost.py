from catboost import CatBoostClassifier
import numpy as np

class CustomCatBoostClassifier(CatBoostClassifier):
    def predict_proba(self, X):
        # Get the original predicted probabilities
        original_probs = super().predict_proba(X)
        
        # Adjust the positive class probabilities (assuming binary classification)
        adjusted_probs = original_probs.copy()
        adjusted_probs[:, 1] = np.where(X['number_of_defaults'] < 1, original_probs[:, 1] - 0.015, original_probs[:, 1])
        adjusted_probs[:, 1] = np.where(X['interest_rate'] <= 13, adjusted_probs[:, 1] + 0.1, adjusted_probs[:, 1])
        adjusted_probs[:, 1] = np.where(X['number_of_defaults'] >= 2, adjusted_probs[:, 1] + 0.025, adjusted_probs[:, 1])

        # Ensure the probabilities are still valid (between 0 and 1)
        adjusted_probs[:, 1] = np.clip(adjusted_probs[:, 1], 0, 1)
        
        return adjusted_probs
