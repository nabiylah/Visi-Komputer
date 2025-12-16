# tugas3_decisiontree.py
import pandas as pd
import numpy as np
from math import log2
from pprint import pformat
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib
matplotlib.use('Agg')  # non-interactive backend untuk server
import matplotlib.pyplot as plt
from flask import render_template, request

# ============================================
# DATASET PLAY GOLF
# ============================================

data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny',
                'Sunny','Rainy','Sunny','Overcast','Overcast','Rainy'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild',
                    'Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High',
                 'Normal','Normal','Normal','High','Normal','High'],
    'Windy': ['False','False','True','False','False','True','True','False',
              'False','False','True','True','False','True'],
    'PlayGolf': ['No','No','Yes','Yes','Yes','No','Yes','No',
                 'Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

# ============================================
# ENTROPY
# ============================================

def entropy(col):
    elements, counts = np.unique(col, return_counts=True)
    ent = 0
    for i in range(len(elements)):
        p = counts[i] / np.sum(counts)
        ent -= p * log2(p)
    return round(ent, 3)

# ============================================
# INFORMATION GAIN
# ============================================

def info_gain(data, split_attribute, target="PlayGolf"):
    total_entropy = entropy(data[target])
    vals, counts = np.unique(data[split_attribute], return_counts=True)

    weighted_entropy = 0
    for i in range(len(vals)):
        subset = data[data[split_attribute] == vals[i]]
        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(subset[target])

    return round(total_entropy - weighted_entropy, 3)

# ============================================
# ID3 TREE (MANUAL)
# ============================================

def Id3(data, originaldata, features, target="PlayGolf", parent_class=None):
    if len(np.unique(data[target])) <= 1:
        return np.unique(data[target])[0]
    elif len(data) == 0:
        return np.unique(originaldata[target])[
            np.argmax(np.unique(originaldata[target], return_counts=True)[1])
        ]
    elif len(features) == 0:
        return parent_class
    else:
        parent_class = np.unique(data[target])[
            np.argmax(np.unique(data[target], return_counts=True)[1])
        ]
        gains = [info_gain(data, f, target) for f in features]
        best_index = np.argmax(gains)
        best_feature = features[best_index]
        tree_dict = {best_feature: {}}
        remaining = [f for f in features if f != best_feature]
        for v in np.unique(data[best_feature]):
            sub = data[data[best_feature] == v]
            subtree = Id3(sub, data, remaining, target, parent_class)
            tree_dict[best_feature][v] = subtree
    return tree_dict

# ============================================
# PREDIKSI MANUAL DARI TREE
# ============================================

def predict_manual(tree_dict, sample):
    if not isinstance(tree_dict, dict):
        return tree_dict
    root = list(tree_dict.keys())[0]
    value = sample[root]
    if value in tree_dict[root]:
        return predict_manual(tree_dict[root][value], sample)
    else:
        return None  # fallback jika value tidak ada

# ============================================
# VISUALISASI SKLEARN → simpan .png
# ============================================

def generate_tree_png():
    le = LabelEncoder()
    df_enc = df.apply(le.fit_transform)
    X = df_enc.drop(columns=["PlayGolf"])
    y = df_enc["PlayGolf"]
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X, y)
    plt.figure(figsize=(10,6))
    tree.plot_tree(clf, feature_names=list(X.columns), class_names=['No','Yes'], filled=True)
    save_path = "static/uploads/tree_tugas3.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

# ============================================
# PAGE HANDLER
# ============================================

def tugas3_page():
    features = list(df.columns[:-1])
    manual_tree = Id3(df, df, features)
    tree_text = pformat(manual_tree, indent=2)
    img_path = generate_tree_png()

    user_result = None
    input_data = {}

    if request.method == "POST":
        input_data = {
            "Outlook": request.form["Outlook"],
            "Temperature": request.form["Temperature"],
            "Humidity": request.form["Humidity"],
            "Windy": request.form["Windy"]
        }
        user_result = predict_manual(manual_tree, input_data)

    return render_template(
        "tugas3_decisiontree.html",
        df=df.to_html(classes="table table-bordered text-start"),
        tree_text=tree_text,
        img_path=img_path,
        user_result=user_result,
        input_data=input_data
    )
