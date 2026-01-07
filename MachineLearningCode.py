"""
Machine Learning code for Medical Premium Price Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class MachineLearningCode:
    def __init__(self):
        """
        Initialize ML class, load dataset
        """
        # Load dataset (adjust path if needed)
        self.df = pd.read_csv("Medicalpremium.csv")

        # Model and data placeholders
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def train(self):
        """
        Train the regression model
        """
        # Features (all columns except PremiumPrice)
        X = self.df.drop("PremiumPrice", axis=1)
        # Target (PremiumPrice)
        y = self.df["PremiumPrice"]

        # Train-test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Random Forest Regressor
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.x_train, self.y_train)

    def predict(self, age, diabetes, bp, transplants, chronic,
                height, weight, allergies, cancer_history, surgeries):
        """
        Predict premium price given patient features
        """
        # Prepare input array
        input_array = np.array([[age, diabetes, bp, transplants, chronic,
                                 height, weight, allergies, cancer_history, surgeries]])

        # Prediction
        prediction = self.model.predict(input_array)[0]

        # Accuracy (R² score on test set)
        accuracy = self.model.score(self.x_test, self.y_test)

        return [round(prediction, 2), accuracy]

    def save_data(self):
        """
        Save dataset back to CSV (optional)
        """
        self.df.to_csv("data/Medicalpremium.csv", index=False)
