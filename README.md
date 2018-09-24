Traditional Chinese Medicine Disease and Herb Prediction (抓药砖家)
The data is intentionally stripped. Please contact the author to obtain needed data.

# Introduction
Traditional Chinese Medicine (TCM) is widely practiced in China and nearby countries in Asia. TCM physicians observe symptoms in patients and prescribe herbal medicine according to the disease profile they identified. Because the diagnosis is highly empirical and personalized, knowledge retrieval of TCM is difficult without years of training.

Thanks to development of probabilistic model and availability of large-scale TCM electronic medical records (EMR), it is now possible to develop tools for discovering of knowledge from TCM EMR. Here, EMR can be represented by a tuple of diseases, symptoms and herbs. 抓药砖家 is such a web-based software tool for TCM disease and herb prediction to help
1) The training of new TCM doctors
2) Patients understand their symptoms, diseases, and prescriptions

Given a set of observed symptoms, 抓药砖家 is able to return a probability distribution of different diseases and herbs in terms of a color map for better visualization. It also has autocomplete and quick look-up features that provide a user-friendly way to interact with the model. The software tool will be based upon a paper written by Wang et al [1].

# Installation
To install required packages
```bash
pip install -r requirements.txt
```

The server can be launched locally by
```
cd tcm_flask
python app.py
```
Then go to localhost:5000 and you should be able to play with the web app.

# Usage
## Autocomplete Search
In the top right search field, you can type a string of symptoms separated by comma to begin a prediction. Once you start typing, the auto-complete system will give you suggestions so that you can easily fill the input box. After you finish typing the symptoms, you can click the "望闻问切" button to begin a prediction. It is recommended if you type multiple synonyms of each symptom since our EMR is not very well preprocessed and has a few synonyms.
## Disease and Herb predictions
The disease prediction will be shown on the left panel and the herb prediction will be shown on the right panel. The prediction is reprsented by a word cloud and size of each word reprsents the probability of each disease/herbs. 
## Disease and herb search
You can click on each disease/herb to start a quick Google search of the disease and herb.
# Implementation Details
## TCM Model
`tcm_flask/tcm_model/tcm_model.py` implements the graphical model and the EM algorithm.

We preprocessed the HIS tuple with the script in `pre-processing/clean_and_pickle.py`. We trained the TCMModel with the EM Algorithm as discussed in [1]. There are some inconsistency between the text and formula in [1], we implemented the algorithm according to the formula. To implement the background cluster, we added an additional disease "Background" for all the EMR records while learning the EMR, and we ignored the background when making predictions. The predictions are based on the posterior conditioned on the symptoms. After learning the model, we serialized the trained model into a pickled file and the model is loaded into the webserver as a Singleton via the Flask application context.

## Web App
### Server
We implemented the backend server with Flask.

The backend server is responsible for interacting with our learned TCM Model, rendering the web page, serving static files and providing API backend for the requests initiated by the front-end. The Web App loads the learned TCM Model to the Flask application context as a singleton upon start. For the webpage, the backend serves a static page, and the contents are loaded via Ajax techniques. Thus, the backend also provide endpoints for autocomplete, disease and herb predictions.

#### API
**Auto-complete**
* Endpoint: /\_autocomplete/symptoms
* Method: GET
* Request: {query: "keyword"}
* Return: [symptom1, symptom2, ...]

**Auto-complete**
* Method: POST
* Endpoint: /\_predict\_diseases
* Request: {symptom: [symptom1, symptom2, ...]}
* Response: [{text: disease1, size: probability1}, {text: disease2, size: probability2}, ...]

**Auto-complete**
* Method: POST
* Endpoint: /\_predict\_herbs
* Request: {symptom: [symptom1, symptom2, ...]}
* Response: [{text: herb1, size: probability1}, {text: herb2, size: probability2}, ...]


### Front End
The front end is responsible for user interaction. We implemented the auto-complete directly with jQuery Ajax. The functionality is provided by a jquery-ui plugin.
The word cloud representation is implemented with a d3 plugin d3.wordcloud. The style is provided by bootstrap 4.

# Contribution
Qing Ye: Probabilistic model implementation
Chong Lu: Server-side operation
Xiaolan Ke: Client-side visualization

# Reference
[1] Wang, Sheng, et al. "A conditional probabilistic model for joint analysis of symptoms, diseases, and herbs in traditional Chinese medicine patient records." Bioinformatics and Biomedicine (BIBM), 2016 IEEE International Conference on. IEEE, 2016.

