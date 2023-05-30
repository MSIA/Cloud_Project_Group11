import os
from pathlib import Path
import streamlit as st
import joblib
import pickle
import numpy as np
import input_feature_eng as inpput_FE
import pandas as pd
import boto3
import logging.config
import yaml


logging.config.fileConfig("local.conf")

logger = logging.getLogger("streamlit")

MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME", "msia423-group11-spotify-artifacts")
DATA_BUCKET_NAME = os.getenv("DATA_BUCKET_NAME", "msia423-group11-spotifty-ready-to-rain")
CONFIG_BUCKET_NAME = os.getenv("CONFIG_BUCKET_NAME", "msia423-group11-spotifty")


# Create artifacts directory to keep model files and data files downloaded from S3 bucket
artifacts = Path() / "artifacts"
artifacts.mkdir(exist_ok=True)

# Download files from S3
def download_s3(bucket_name: str, object_key: str, local_file_path: Path):
    s3 = boto3.client("s3")
    logger.info(f"Fetching Key: {object_key} from S3 Bucket: {bucket_name}")
    try:
        s3.download_file(bucket_name, object_key, str(local_file_path))
        logger.debug(f"File downloaded successfully to {local_file_path}")
    except Exception as e:
        logger.error(f"Error downloading file: {e}")

@st.cache_data
def load_config(s3_key, config_file):
    download_s3(CONFIG_BUCKET_NAME, s3_key, config_file)
    with config_file.open() as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

@st.cache_data 
def load_data(s3_key, data_file):
    download_s3(DATA_BUCKET_NAME, s3_key, data_file)
    spotify_df = pd.read_csv(data_file)
    return spotify_df

@st.cache_resource
def load_model(s3_key, model_file):
    # Download files from S3
    download_s3(MODEL_BUCKET_NAME, s3_key, model_file)
    # Load the model from the pickle file
    with open(model_file, "rb") as file:
        loaded_model = pickle.load(file)

    return loaded_model

# Check available model versions in S3 bucket and return them as a list
@st.cache_resource
def load_model_versions(s3_models_path):
    model_versions = []
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=MODEL_BUCKET_NAME, Prefix=s3_models_path)
    if 'Contents' in response:
        for obj in response['Contents']:
            file_key = obj['Key']
            # Print or process the file key as needed
            file_name = file_key.split("/")[-1]
            model_versions.append(file_name.split(".")[0])

    return model_versions

def slider_values(series) -> tuple[float, float, float]:
    return (
        float(series.min()),
        float(series.max()),
        float(series.mean()),
    )

def select_categories(series) -> list:
    """Select all possible values from a given categotical column so that users can have 
    entire options to choose"""
    series = list(series)
    # distinct categories in the series
    choices = list(set(series))

    return choices


def main():
    # Create the application title and description
    st.title("Catch the Trend and Predict Popularity of Spotify Songs")
    # st.image("streamlit_UI/fire_emoji.png", use_column_width=True)
    st.write("This app predicts the popularity of a song given its attributes")

    # Allow users to select multiple models
    st.subheader("Model Selection")

  
    # location of the folder which stores all models in S3 bucket
    s3_models_path = "artifacts/models"
    # Find available model versions in artifacts dir
    available_models = load_model_versions(s3_models_path)
    logger.debug("available_models in the folder: ", available_models)

    # set default model version: best_model
    model_version = "best_model"
    # Create a dropdown to select the model
    model_version = st.selectbox("Select Model", list(available_models))


    st.image("spotify_pic.png", use_column_width=True)


    # location of the data and config file in S3 bucket
    spotify_data_s3key = "cleaned_data.csv"
    spotify_config_s3key = "default-config.yaml"
     # location of the model file in S3 bucket
    model_filename = model_version + ".pkl"
    spotify_model_s3key = s3_models_path + "/" + model_filename

    

    # Establish local path to store the dataset and TMO locations based on selection
    spotify_file = artifacts / "cleaned_Spotify_data.csv"
    spotify_model_file = artifacts / model_filename
    # Establish local path to store the config yaml file
    spotify_config_file = artifacts  / "default-config.yaml"

    # Load the dataset and TMO into memory
    df = load_data(spotify_data_s3key, spotify_file)
    model = load_model(spotify_model_s3key , spotify_model_file)
    config = load_config(spotify_config_s3key,spotify_config_file)
    # Load the configuration for implementing feature engineering on user input data
    input_feat_config = config.get("stremalit_inference", {})


    # Sidebar inputs for categorical features
    st.sidebar.header("Input Parameters")
    genre = st.sidebar.radio("Genre", select_categories(df["genre"]))
    key = st.sidebar.radio("Key", select_categories(df["key"]))
    mode = st.sidebar.radio("Mode", select_categories(df["mode"]))
    time_sig = st.sidebar.radio("Time Signature", select_categories(df["time_signature"]))

    # Sidebar inputs for numeric features
    acousticness = st.sidebar.slider("Acousticness", *slider_values(df["acousticness"]))
    danceability = st.sidebar.slider("Danceability", *slider_values(df["danceability"]))
    # set default value for duration
    duration = 235122

    # Get input from UI
    energy =  st.sidebar.slider("Energy", *slider_values(df["energy"]))
    instrumentalness = st.sidebar.slider("Instrumentalness", *slider_values(df["instrumentalness"]))
    liveness = st.sidebar.slider("Liveness", *slider_values(df["liveness"]))
    loudness = st.sidebar.slider("Loudness", *slider_values(df["loudness"]))
    speechiness = st.sidebar.slider("Speechiness", *slider_values(df["speechiness"]))
    tempo = st.sidebar.slider("Tempo", *slider_values(df["tempo"]))
    valence =  st.sidebar.slider("Valence", *slider_values(df["valence"]))

    # Prediction based on user input
    input_df = pd.DataFrame({"genre": [genre], "acousticness": [acousticness], "danceability": [danceability], "duration_ms": [duration], "energy": [energy],
                            "instrumentalness": [instrumentalness], "key": [key], "liveness": [liveness], "loudness": [loudness], "mode": [mode], "speechiness":[speechiness],
                            "tempo": [tempo], "time_sig": [time_sig], "valence": [valence] })

    input_df_feat = inpput_FE.input_feature_eng(input_df,input_feat_config)

    # Predict Popularity score
    prediction = model.predict(input_df_feat)
    pop_score = round(prediction[0],4)

    # Display the predicted popularity
    st.subheader("Predicted Popularity:")
    st.markdown(f"<p style='color: orange; font-size: 50px;'>{pop_score}</p>", unsafe_allow_html=True)

    # Create layout columns for positioning
    col1, col2 = st.columns([1,3])

    with col2:
        st.image("fire.png", width=100, use_column_width=False)



if __name__ == "__main__":
    main()