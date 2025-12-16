from flask import Flask, flash, render_template, request
from modules import tugas1_glcm
from modules import tugas2_knn
from modules import tugas3_decisiontree
from modules import tugas4_naivebayes
from flask import Flask, render_template, request, flash
from modules.tugas2_knn import predict_stunting_wasting
from modules.tugas2_knn import evaluation
import os


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.secret_key = 'replace-with-some-secret'  # untuk flash message
CSV_PATH = os.path.join('data', 'stunting_wasting_dataset.csv')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tugas1', methods=['GET', 'POST'])
def tugas1():
    if request.method == 'POST':
        file = request.files['image']
        angle = int(request.form['angle'])
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        result = tugas1_glcm.process(filepath, angle)
        return render_template('tugas1_glcm.html',
                               filename=file.filename,
                               angle=angle,
                               result=result)
    return render_template('tugas1_glcm.html')

@app.route("/tugas2", methods=["GET", "POST"])
def tugas2_knn():
    result = None

    if request.method == "POST":
        try:
            gender = int(request.form["gender"])
            age = float(request.form["age"])
            height = float(request.form["height"])
            weight = float(request.form["weight"])
            k = int(request.form["k"])

            stunting, wasting = predict_stunting_wasting(
                gender, age, height, weight, k
            )

            result = {
                "stunting": stunting,
                "wasting": wasting
            }

        except Exception as e:
            flash(str(e))

    return render_template("tugas2_knn.html", result=result, evaluation=evaluation)

@app.route("/tugas3", methods=["GET", "POST"])
def tugas3():
    return tugas3_decisiontree.tugas3_page()

@app.route("/tugas4", methods=["GET", "POST"])
def tugas4():
    from flask import request, render_template

    if request.method == "POST":
        usia = request.form["usia"]
        penghasilan = request.form["penghasilan"]
        promo = request.form["promo"]

        pred = tugas4_naivebayes.predict_naive_bayes(usia, penghasilan, promo)

        return render_template(
            "tugas4_naivebayes.html",
            show_result=True,
            result=pred,
            usia=usia,          
            penghasilan=penghasilan,
            promo=promo,
            df=tugas4_naivebayes.get_dataset_html()
        )

    return tugas4_naivebayes.tugas4_page()

if __name__ == '__main__':
    app.run(debug=True)
