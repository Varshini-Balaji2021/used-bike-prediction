import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    form_val = list(request.form.values())
    form_val1 = []
    form_val1.append(form_val[2])
    form_val1.append(form_val[4])
    form_val1.append(form_val[5])
    print(form_val1)
    int_features = [float(x) for x in form_val1]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Used bike prediction :{}'.format(output))
if __name__ == "__main__":
    app.run(debug=True)

