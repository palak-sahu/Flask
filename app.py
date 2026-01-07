"""
Flask app for predicting medical insurance premium price using ML model
"""

from flask import Flask, render_template, request
from MachineLearningCode import MachineLearningCode

# app initialization
app = Flask(__name__)
print("Flask app initialized")

# Initialize and Train the model ONCE when the app starts
MLC = MachineLearningCode()
MLC.train()

# route definitions
@app.route('/')
@app.route('/home')
def home():
    """
    Home page route
    """
    return render_template("home.html")

# Changed to methods=['POST'] only for better security
@app.route('/predict', methods=['POST'])
def predictandoutput():
    """
    Prediction route
    """
    print("Inside predictandoutput function")

    # Collect form inputs
    # Note: request.form returns strings like "1", so we convert to int/float
    try:
        age = float(request.form["Age"])
        diabetes = int(request.form["Diabetes"])
        bp = int(request.form["BloodPressureProblems"])
        transplants = int(request.form["AnyTransplants"])
        chronic = int(request.form["AnyChronicDiseases"])
        height = float(request.form["Height"])
        weight = float(request.form["Weight"])
        allergies = int(request.form["KnownAllergies"])
        cancer_history = int(request.form["HistoryOfCancerInFamily"])
        surgeries = int(request.form["NumberOfMajorSurgeries"])

        # Predict premium using the already trained MLC object
        result = MLC.predict(
            age, diabetes, bp, transplants, chronic,
            height, weight, allergies, cancer_history, surgeries
        )

        return render_template(
            "result.html",
            Predicted_premium=result[0],
            Accuracy_of_prediction=int(result[1] * 100)
        )
    except Exception as e:
        return f"Error processing prediction: {str(e)}"

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/prediction')
def prediction():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)