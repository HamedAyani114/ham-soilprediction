import pickle
from flask import Flask, render_template, request
import numpy as np
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    acc_path = os.path.join('models', 'accGNB')
    acc = pickle.load(open(acc_path, 'rb'))
    if request.method == "POST":
        # request all the input fields N	P	K	temperature	humidity	ph	rainfall
        N = request.form['N']
        P = request.form['P']
        K = request.form['K']
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        val = np.array([N, P, K, temperature, humidity, ph, rainfall])
        datain = [np.array(val)]

        scalar_path = os.path.join('models', 'scalerData')
        scalar = pickle.load(open(scalar_path, 'rb'))


        model_path = os.path.join('models', 'modelsoilgnb_norm.sav')
        model = pickle.load(open(model_path, 'rb'))

        acc_path = os.path.join('models', 'accGNB')
        acc = pickle.load(open(acc_path, 'rb'))

        final_features = scalar.transform(datain)
        res = model.predict(final_features)

        return render_template('index.html', result=res[0], acc=acc)
    return render_template('index.html', acc=acc)



# run application
if __name__ == "__main__":
    app.run()
