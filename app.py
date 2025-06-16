import os
import pandas as pd
from scipy.sparse import csr_matrix
import pickle
import flask
from flask import jsonify
from huggingface_hub import login, hf_hub_download
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
login(token=HF_TOKEN)

# Model
REPO_ID = "mmassonn/Recommandation_model"
MODEL_FILE_NAME = "recommender.model"
model_path = hf_hub_download(repo_id=REPO_ID,filename=MODEL_FILE_NAME)

# Dataset
REPO_ID_DATASET = "mmassonn/recommandation"
DATA_FILE_NAME = "clicks.csv"
clicks_path = hf_hub_download(repo_id=REPO_ID_DATASET,filename=DATA_FILE_NAME)
clicks = pd.read_csv(clicks_path)

def compute_interaction_matrix(clicks):
    interactions = clicks.groupby(['user_id', 'article_id']).size().reset_index(name='count')
    csr_item_user = csr_matrix((interactions['count'].astype(float),
                                (interactions['article_id'],
                                 interactions['user_id'])))
    csr_user_item = csr_matrix((interactions['count'].astype(float),
                                (interactions['user_id'],
                                 interactions['article_id'])))
    return csr_item_user, csr_user_item


def get_cf_reco(clicks, userID, csr_item_user, csr_user_item, model_path, n_reco=5):
    with open(MODEL_PATH, 'rb') as filehandle:
        model = pickle.load(filehandle)
    recommendations_list = []
    recommendations = model.recommend(userID, csr_user_item[userID], N=n_reco, filter_already_liked_items=True)
    return recommendations[0].tolist()


csr_item_user, csr_user_item = compute_interaction_matrix(clicks)

app = flask.Flask(__name__)

@app.route("/")
def home():
    return "Welcome on the recommendation API ! "

@app.route("/get_recommendation/<id>", methods=["POST", "GET"])
def get_recommendation(id):

    recommendations = get_cf_reco(clicks, int(id), csr_item_user, csr_user_item, model_path=MODEL_PATH, n_reco=5, train=False)
    data = {
            "user" : id,
            "recommendations" : recommendations,
        }
    return jsonify(data)
