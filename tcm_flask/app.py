from flask import Flask, render_template, redirect, url_for, request, session
from flask import jsonify, g
from tcm_model.tcm_model import TCMModel

app = Flask(__name__)

def get_tcm_model():
    tcm = getattr(g, '_tcm_model', None)
    if tcm is None:
        tcm = TCMModel.from_saved('../resource/tcm_model_3.pickle')
    return tcm

def get_symptoms_list():
    symptoms = getattr(g, '_symptoms', None)
    if symptoms is None:
        tcm = get_tcm_model()
        symptoms = tcm.emr.symptoms
    return symptoms

# app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

#index page, display home page and search bar
@app.route('/')
def index():
    return render_template('index.html')

#result page, show resulting probability distribution and visualization
@app.route('/sresult', methods=['GET', 'POST'])
def sresult():
    key_word = request.form.get('search_box')
    if(key_word):
        # dic = autocom.parseDataFile()
        # words = autocom.retrieve(dic, key_word)
        # return jsonify(json_list=words)

        #TO DO
        #calculate the probability distribution based on prediction models API

        return render_template('sresult.html', words = words)
    return "Shouldn't reach here..."

# auto-complete api, given a chinese character, return a list of possible words in json format
@app.route('/_autocomplete/symptoms')
def autocomplete():
    start = request.args.get('query', None)
    symptoms = get_symptoms_list()
    symptoms = [s for s in symptoms if s.startswith(start)][:10]
    return jsonify(symptoms)

@app.route('/_predict_diseases', methods=["POST"])
def predict_diseases():
    symptoms = request.get_json()['symptoms']
    symptoms = [s for s in symptoms if s]
    tcm = get_tcm_model()
    diseases, prob_dist = tcm.predict_disease(symptoms, 20)
    response = [{'text':x, 'size':y + 0.000001} for x, y in zip(diseases, prob_dist)]
    return jsonify(response)

@app.route('/_predict_herbs', methods=["POST"])
def predict_herbs():
    symptoms = request.get_json()['symptoms']
    symptoms = [s for s in symptoms if s]
    tcm = get_tcm_model()
    herbs, prob_dist = tcm.predict_herbs(symptoms, 20)
    response = [{'text':x, 'size':y + 0.000001} for x, y in zip(herbs, prob_dist)]
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
