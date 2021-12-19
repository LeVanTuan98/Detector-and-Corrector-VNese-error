from flask import Flask, render_template, request
from model import *
import clean_text as cleaner

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

@app.route('/check', methods=['POST'])
def checking():
    # html -> py
    if request.method == 'POST':
        raw_doc = request.form['input']
        print('raw: ', raw_doc)
        doc = cleaner.clean_html(raw_doc)
        checked_doc = predict(model, doc)
        # checked_doc = [('aa', 1, ('a', 'b', 'c')), ('bb', 0, ())]
        print('check: ', checked_doc)
    # py -> html
    return render_template('checking.html', words=checked_doc)


if __name__ == '__main__':
    global model
    model = load_model()
    app.run(debug=True)
