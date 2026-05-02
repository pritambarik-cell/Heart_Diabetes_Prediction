import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class DiabetesPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    @staticmethod
    def load_and_preprocess_data(file_path):
        """Load and preprocess the diabetes dataset"""
        try:
            data = pd.read_csv(file_path)
            print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")

            # Handle missing values
            if data.isnull().sum().any():
                data = data.fillna(data.mean())

            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    @staticmethod
    def create_visualizations(data, target_column='Outcome'):
        """Create exploratory data analysis visualizations"""
        if data is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Distribution of target variable
        outcome_counts = data[target_column].value_counts()
        axes[0, 0].pie(outcome_counts.values, labels=['No Diabetes', 'Diabetes'],
                       autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Diabetes Distribution')

        # Age distribution by diabetes outcome
        data.boxplot(column='Age', by=target_column, ax=axes[0, 1])
        axes[0, 1].set_title('Age Distribution by Diabetes Outcome')

        # Glucose levels by outcome
        data.boxplot(column='Glucose', by=target_column, ax=axes[1, 0])
        axes[1, 0].set_title('Glucose Levels by Diabetes Outcome')

        # Correlation heatmap
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Feature Correlation Heatmap')

        plt.tight_layout()
        plt.show()

    def train_model(self, data, target_column='Outcome'):
        """Train the Random Forest model"""
        if data is None:
            print("No data available for training")
            return False

        try:
            # Prepare features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train the model
            self.model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = self.model.predict(X_test_scaled)

            # Calculate and display metrics
            model_accuracy = accuracy_score(y_test, y_pred)
            print(f"Model trained successfully!")
            print(f"Accuracy: {model_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No Diabetes', 'Diabetes'],
                        yticklabels=['No Diabetes', 'Diabetes'])
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()

            self.is_trained = True
            return True

        except Exception as e:
            print(f"Error during training: {e}")
            return False

    def predict_diabetes(self, features):
        """Predict diabetes probability for new data"""
        if not self.is_trained:
            print("Model not trained yet. Please train the model first.")
            return None

        try:
            # Ensure features are in correct format
            if isinstance(features, list):
                features = np.array(features).reshape(1, -1)

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]

            # Convert numpy types to native Python types
            prediction = int(prediction)  # Convert numpy int to Python int
            probability = float(probability[1])  # Convert numpy float to Python float

            return {
                'prediction': prediction,
                'probability': probability,
                'risk_level': self.get_risk_level(probability)
            }

        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    @staticmethod
    def get_risk_level(probability):
        """Determine risk level based on probability"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"

    def feature_importance(self):
        """Display feature importance"""
        if not self.is_trained:
            print("Model not trained yet.")
            return

        # Get feature importance
        importance = self.model.feature_importances_
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        # Create feature importance plot
        feature_imp_df = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_imp_df['feature'], feature_imp_df['importance'])
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the diabetes predictor"""
    # Initialize predictor
    predictor = DiabetesPredictor()

    # Load data with correct path
    data = predictor.load_and_preprocess_data(
        r'C:\Users\KIIT0001\PycharmProjects\Medicalpredictor\datasets\diabetes.csv')

    if data is not None:
        # Create visualizations
        predictor.create_visualizations(data)

        # Train model
        predictor.train_model(data)

        # Show feature importance
        predictor.feature_importance()

        # Example prediction
        example_features = [2, 120, 70, 20, 79, 25.0, 0.5, 30]
        result = predictor.predict_diabetes(example_features)

        if result:
            print(f"\nPrediction Result:")
            print(f"Diabetes: {'Yes' if result['prediction'] == 1 else 'No'}")
            print(f"Probability: {result['probability']:.4f}")
            print(f"Risk Level: {result['risk_level']}")


if __name__ == "__main__":
    main()