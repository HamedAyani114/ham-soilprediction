import pickle
from flask import Flask, render_template, request
import numpy as np
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
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

        final_features = [np.array(val)]
        model_path = os.path.join('models', 'modelsoilgnb.sav')
        model = pickle.load(open(model_path, 'rb'))
        res = model.predict(final_features)

        return render_template('index.html', result=res)
    return render_template('index.html')



# run application
if __name__ == "__main__":
    app.run()
