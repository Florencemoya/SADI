import numpy as np
from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import LabelEncoder



app = Flask(__name__, template_folder='templates')


# Charger le modèle pré-entraîné
model = joblib.load('model.pkl')

# Charger les encodeurs pré-entraînés
le_etio = joblib.load('etio_encoder.pkl')
le_cv = joblib.load('cv_encoder.pkl')
# Créer des variables globales pour stocker les encodeurs
#le_sexe = LabelEncoder()
#le_sexe.classes_ = np.array([0, 1])  # Correspond aux classes 'F' et 'H'

@app.route('/',methods=['GET'])
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['Age'])
    esterman = float(request.form['Esterman'])
    st = float(request.form['ST'])
    dt_centrale = float(request.form['DT_centrale'])
    dt_quadsupd = float(request.form['DT_quadsupd'])
    dt_quadsupg = float(request.form['DT_quadsupg'])
    dt_quadinfg = float(request.form['DT_quadinfg'])
    dt_quadinfd = float(request.form['DT_quadinfd'])
    dt_dis_Centrale = float(request.form['DT_dis_Centrale'])
    dt_dis_quadsupg = float(request.form['DT_dis_quadsupg'])
    dt_dis_quadsupd = float(request.form['DT_dis_quadsupd'])
    dt_dis_quadinfg = float(request.form['DT_dis_quadinfg'])
    dt_dis_quadinfd = float(request.form['DT_dis_quadinfd'])

    # Récupérer les valeurs sélectionnées par l'utilisateur depuis le formulaire
    sexe = request.form['sexe']
    etio = request.form['etio']
    cv = request.form['cv']

    sexe_encoded = 0 if sexe == 'F' else 1
    #sexe_encoded = le_sexe.transform([sexe])[0]
    etio_encoded = le_etio.transform([etio])[0]
    cv_encoded = le_cv.transform([cv])[0]

    # ... (autres parties du code)

    # Ajouter les valeurs encodées à features pour la prédiction
    features = np.array([dt_quadsupg, dt_quadinfg, dt_quadsupd, dt_quadinfd, dt_dis_quadsupg,
        dt_dis_quadinfg, dt_dis_quadsupd, dt_dis_quadinfd, st, dt_centrale,
        dt_dis_Centrale, sexe_encoded, etio_encoded, cv_encoded, esterman,age]).reshape(1, -1)

    # Prédiction de la décision de conduite en utilisant le modèle
    decision_conduite = model.predict(features)[0]

    return render_template('result.html', decision_conduite=decision_conduite)

if __name__ == '__main__':
    app.run(debug=True)