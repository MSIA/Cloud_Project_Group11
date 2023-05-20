from pathlib import Path
import logging
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def save_figures(data: pd.DataFrame, filepath: Path) -> list[Path]:
    """_summary_

    Args:
        data: Dataframe containing data for generating figures
        filepath: directory in which to save generated figures

    Returns:
        a list of file paths of saved figures
    """
    logger.info("Visualizing the data")
    figs = []
    for feat in data.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        try:
            if feat == "artist_name":   # plot distribution for average song popularity of each artist
                artist_popularity = data.groupby("artist_name")["popularity"].mean().sort_values(ascending=False)
                artist_popularity = pd.DataFrame(artist_popularity)
                sns.histplot(artist_popularity["popularity"], ax=ax)
                ax.set_title("Distribution for average song popularity of each artist")
                ax.set_xlabel("Avaerage popularity score")
                ax.set_ylabel("Number of artists")
            
            elif data[feat].dtype.kind in 'biufc':  # plot distribution for all numeric columns
                sns.histplot(data[feat].values)
                ax.set_xlabel(" ".join(feat.split("_")).capitalize())
                ax.set_ylabel("Number of observations")
            
            else: 
                continue

        except KeyError as e:
            logger.error(
                "The class column is missing for visualizing two cloud types %s", e
            )
            sys.exit(1)


        try:
            if feat == "artist_name":
                fig_path = filepath / "avg_popularity_per_artists.png"
            else:
                fig_path = filepath / f"{feat}.png"
            fig.savefig(fig_path)
        except FileNotFoundError as e:
            logger.error(
                "The directory path in which to save figures is not found %s", e
            )
            sys.exit(1)
        figs.append(fig_path)
        plt.close(fig)
    logger.info(f"Successfully saved {len(figs)} figures to {filepath}.")
    return figs