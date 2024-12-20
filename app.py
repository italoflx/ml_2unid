from flask import Flask, request, render_template
import joblib
import numpy as np

# Carregar o modelo salvo
model_path = 'best_model_pipeline.pkl'
model = joblib.load(model_path)

# Criar aplicação Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['ApplicantIncome']),
            float(request.form['LoanAmount']),
            int(request.form['Gender']),
            int(request.form['Married']),
            int(request.form['Dependents']),
            int(request.form['Education']),
            int(request.form['Self_Employed'])
        ]

        # Fazer a previsão
        prediction = model.predict([features])[0]
        prob = model.predict_proba([features])[0]

        # Traduzir o resultado
        status = "Aprovado" if prediction == 1 else "Negado"
        prob_percent = round(prob[prediction] * 100, 2)

        return render_template('result.html', status=status, probability=prob_percent)
    except Exception as e:
        return f"Ocorreu um erro: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)