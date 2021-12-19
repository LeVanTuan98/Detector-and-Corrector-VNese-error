from flask import Flask, render_template, request
import model

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def predict():
    output = 'd'
    if request.method == 'POST':
        hrs = request.form['hrs']
        output = model.prediction(hrs)
        # print(pred)
    return render_template('index.html', pred=output)

# @app.route('/sub', methods=['POST'])
# def submit():
#     # html -> py
#     user_name = '<error>'
#     if request.method == 'POST':
#         user_name = request.form['username']
#     # py -> html
#     return render_template('sub.html', n=user_name)

if __name__ == '__main__':
    app.run(debug=True)
