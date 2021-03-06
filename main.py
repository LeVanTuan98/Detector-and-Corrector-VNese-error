from flask import Flask, render_template, request
# from model import *
import clean_text as cleaner
# from correction import *
from hardmasked_predict import *

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')
    

# @app.route("/", methods=['GET', 'POST'])

# def predict():
#     output = 'd'
#     if request.method == 'POST':
#         hrs = request.form['hrs']
#         output = model.prediction(hrs)
#         # print(pred)
#     return render_template('index.html')

@app.route('/check', methods=['POST', 'GET'])
def checking():
    # html -> py
    if request.method == 'POST':
        raw_doc = request.form['input']
        print('raw: ', raw_doc)
        doc = cleaner.clean_html(raw_doc)
        global checked_doc
        checked_doc = predict(doc, model)
        # checked_doc = model(doc)
        
        # checked_doc = [('aa', 1, ('a', 'b', 'c')), ('bb', 0, ())]
        print('check: ', checked_doc)
        num_words = 2
    else: 
        num_words = int(request.args.get('n'))
    # py -> html
    return render_template('checking.html', words=checked_doc, n=num_words)


# if __name__ == '__main__':
#     # global model
model = load_model()
app.run(debug=False)
