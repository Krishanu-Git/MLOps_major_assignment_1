# test.py
import logging
import sys
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Loading Olivetti faces dataset...")
        data = fetch_olivetti_faces()
        X = data.images
        y = data.target
        n_samples = len(X)
        logger.info("Dataset loaded: %d samples, image shape %s", n_samples, X[0].shape if n_samples else ())
        X = X.reshape((n_samples, -1))
        logger.debug("Reshaped X to %s", X.shape)

        logger.info("Creating train/test split (test_size=0.3, random_state=42, stratify=y)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        logger.info("Split complete: train=%d, test=%d", len(X_train), len(X_test))

        # Load Model
        logger.info("Loading model from 'savedmodel.pth'...")
        try:
            clf = joblib.load('savedmodel.pth')
        except Exception:
            logger.exception("Failed to load model file 'savedmodel.pth'. Make sure the file exists and is a valid joblib file.")
            raise

        # Predict
        logger.info("Making predictions on test set...")
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info("Test Accuracy: %.2f%%", acc * 100)

    except Exception:
        logger.exception("An unexpected error occurred.")
        sys.exit(1)

if __name__ == "__main__":
    main()