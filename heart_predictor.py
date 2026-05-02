import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')


class HeartDiseasePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    @staticmethod
    def load_and_preprocess_data(file_path):
        """Load and preprocess the heart disease dataset"""
        try:
            # Get the project root directory and construct full path
            project_root = os.path.dirname(os.path.abspath(__file__))
            datasets_path = os.path.join(project_root, 'datasets')
            full_path = os.path.join(datasets_path, 'heart.csv')

            print(f"📊 Loading Heart Disease Dataset...")
            print(f"Looking for file at: {full_path}")

            # Check if file exists
            if not os.path.exists(full_path):
                print(f"❌ File not found: {full_path}")
                print("Please make sure 'heart.csv' exists in the datasets folder")
                return None

            df = pd.read_csv(full_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📁 Shape: {df.shape}")
            print(f"🔍 Columns: {list(df.columns)}")

            # Handle missing values if any
            if df.isnull().sum().any():
                print("🔄 Handling missing values...")
                df = df.fillna(df.mean())

            return df

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def train_model(self, df, target_column='target'):
        """Train the heart disease prediction model"""
        if df is None:
            print("❌ No data available for training")
            return False

        try:
            print("🔄 Preparing data for training...")
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            print("📊 Scaling features...")
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            print("🤖 Training Random Forest model...")
            # Train the model
            self.model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = self.model.predict(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            print(f"✅ Model trained successfully!")
            print(f"🎯 Accuracy: {accuracy:.4f}")
            print("\n📋 Classification Report:")
            print(classification_report(y_test, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No Heart Disease', 'Heart Disease'],
                        yticklabels=['No Heart Disease', 'Heart Disease'])
            plt.title('Heart Disease - Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()

            self.is_trained = True
            return True

        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False

    def predict_heart_disease(self, features):
        """Predict heart disease for new patient data"""
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
                'risk_level': self._get_risk_level(probability)
            }

        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    @staticmethod
    def _get_risk_level(probability):
        """Determine risk level based on probability"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"

    def feature_importance(self, feature_names=None):
        """Display feature importance"""
        if not self.is_trained:
            print("❌ Model not trained yet.")
            return

        # Get feature importance
        importance = self.model.feature_importances_

        # Default feature names for heart dataset
        if feature_names is None:
            feature_names = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]

        # Create feature importance plot
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(feature_imp_df['feature'], feature_imp_df['importance'])
        plt.title('Heart Disease - Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()


def demo_heart_prediction():
    """Demo function to test the heart disease predictor"""
    print("❤️ Heart Disease Prediction Demo")
    print("=" * 40)

    # Initialize predictor
    predictor = HeartDiseasePredictor()

    # Load data - FIXED PATH
    df = predictor.load_and_preprocess_data('datasets/heart.csv')

    if df is not None:
        # Train model
        success = predictor.train_model(df)

        if success:
            # Show feature importance
            predictor.feature_importance()

            # Example prediction - sample patient data
            print("\n🧪 Example Prediction:")
            # Example features: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            example_patient = [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]

            result = predictor.predict_heart_disease(example_patient)

            if result:
                print(f"🎯 Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}")
                print(f"📊 Probability: {result['probability']:.4f}")
                print(f"⚠️ Risk Level: {result['risk_level']}")

                # Interpretation
                if result['prediction'] == 1:
                    print("💡 Recommendation: Consult a cardiologist for further evaluation")
                else:
                    print("💡 Recommendation: Maintain healthy lifestyle with regular checkups")


def check_heart_dataset():
    """Check if heart dataset exists and show info"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    datasets_path = os.path.join(project_root, 'datasets')
    heart_path = os.path.join(datasets_path, 'heart.csv')

    print("🔍 Checking project structure...")
    print(f"Project root: {project_root}")
    print(f"Datasets folder exists: {os.path.exists(datasets_path)}")
    print(f"Heart CSV exists: {os.path.exists(heart_path)}")

    if os.path.exists(datasets_path):
        print("📁 Files in datasets folder:")
        for file in os.listdir(datasets_path):
            print(f"   - {file}")

    return os.path.exists(heart_path)


if __name__ == "__main__":
    # First check if dataset exists
    if not check_heart_dataset():
        print("\n❌ ERROR: Please make sure 'heart.csv' exists in the datasets folder!")
        print("💡 The file should be at: datasets/heart.csv")
    else:
        # Run the demo
        demo_heart_prediction()