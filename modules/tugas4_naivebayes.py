# tugas4_naivebayes.py

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

# Dataset
data = [
    ("Muda",   "Rendah",  "Ya",    "Tidak"),
    ("Muda",   "Rendah",  "Tidak", "Tidak"),
    ("Muda",   "Sedang",  "Tidak", "Ya"),
    ("Tengah", "Rendah",  "Ya",    "Ya"),
    ("Tengah", "Sedang",  "Tidak", "Ya"),
    ("Tengah", "Tinggi",  "Ya",    "Ya"),
    ("Tua",    "Sedang",  "Tidak", "Tidak"),
    ("Tua",    "Tinggi",  "Tidak", "Ya"),
    ("Muda",   "Tinggi",  "Ya",    "Ya"),
    ("Tua",    "Rendah",  "Tidak", "Tidak"),
    ("Tengah", "Tinggi",  "Tidak", "Ya"),
    ("Muda",   "Sedang",  "Ya",    "Ya"),
]

X = [row[:-1] for row in data]  
y = [row[-1] for row in data]   

# Encode fitur kategori
encoders = [LabelEncoder() for _ in range(len(X[0]))]

X_encoded = np.column_stack([
    encoders[i].fit_transform([row[i] for row in X])
    for i in range(len(X[0]))
])

y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)

# Model
model = CategoricalNB()
model.fit(X_encoded, y_encoded)


def predict_naive_bayes(usia, penghasilan, promo):
    test = [(usia, penghasilan, promo)]

    test_encoded = np.column_stack([
        encoders[i].transform([row[i] for row in test])
        for i in range(len(test[0]))
    ])

    pred = model.predict(test_encoded)[0]
    result = y_encoder.inverse_transform([pred])[0]

    return result


def get_dataset_html():
    html = """
    <table class="table table-striped table-bordered">
        <thead class="table-primary">
            <tr>
                <th>Usia</th>
                <th>Penghasilan</th>
                <th>Promo</th>
                <th>Beli</th>
            </tr>
        </thead>
        <tbody>
    """

    for row in data:
        html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td></tr>"

    html += "</tbody></table>"
    return html


def tugas4_page():
    from flask import render_template
    return render_template("tugas4_naivebayes.html", show_result=False, result=None, df=get_dataset_html())
