import argparse
import datetime
import logging.config
from pathlib import Path
import yaml

# import src.acquire_data as ad
import src.analysis as eda
# import src.aws_utils as aws
import src.create_dataset as cd
import src.evaluate_performance as ep
import src.feature_engineer as fe
import src.score_model as sm
import src.train_models as tm


logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("main")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from spotify data"
    )
    parser.add_argument(
        "--config", default="config/default-config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error("Error while loading configuration from %s", args.config)
        else:
            logger.info("Configuration file loaded from %s", args.config)

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "runs")) / str(now)
    artifacts.mkdir(parents=True)

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # # Acquire data from online repository and save to disk
    # ad.acquire_data(run_config["data_source"], artifacts / "clouds.data")

    # Create structured dataset from raw data and implement basice data cleaning; save to disk
    raw_data_path = "data/SpotifyFeatures.csv"
    cleaned_df = cd.create_cleaned_dataset(raw_data_path)
    cd.save_dataset(cleaned_df, artifacts / "cleaned_Spotify_data.csv")

    # Generate statistics and visualizations for summarizing the data; save to disk
    figures = artifacts / "figures"
    figures.mkdir()
    eda.save_figures(cleaned_df, figures)

    # Enrich dataset with features for model training; save to disk
    features = fe.feature_eng(cleaned_df, config["feature_engineer"])
    cd.save_dataset(features, artifacts / "data_ready_to_train.csv")

    # Split data into train/test set and train model based on config; save each to disk
    tmo, train, test = tm.train_model(features, config["train_model"])
    tm.save_data(train, test, artifacts)
    tm.save_model(tmo, artifacts / "trained_model_object.pkl")

    # Score model on test set; save scores to disk
    model_path = artifacts / "trained_model_object.pkl"
    scores = sm.model_prediction(test, model_path , config["score_model"])
    sm.save_scores(scores, artifacts / "scores.csv")

    # Evaluate model performance metrics; save metrics to disk
    metrics = ep.evaluate_performance(scores, config["evaluate_performance"])
    ep.save_metrics(metrics, artifacts / "metrics.yaml")
    logger.info("All tasks are finsihed successfully; End of the project")

