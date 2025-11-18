# train.py
import sys
import logging
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

def setup_logger(log_file: str = "train.log"):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream handler (stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def main():
    logger = setup_logger()

    try:
        logger.info("Loading dataset (Olivetti faces)...")
        data = fetch_olivetti_faces()
        X = data.images
        y = data.target
        logger.info("Dataset loaded: %d samples, image shape %s", len(X), X.shape[1:])

        # Flatten images: (n_samples, 64, 64) -> (n_samples, 4096)
        n_samples = len(X)
        X = X.reshape((n_samples, -1))
        logger.info("Flattened images to shape %s", X.shape)

        # Split data (70% train, 30% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        logger.info(
            "Split data: train=%d, test=%d (test_size=0.3, stratify=y)",
            len(X_train),
            len(X_test),
        )

        logger.info("Training DecisionTreeClassifier...")
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        logger.info("Training complete.")

        test_acc = clf.score(X_test, y_test)
        logger.info("Test accuracy: %.4f", test_acc)

        save_path = "savedmodel.pth"
        joblib.dump(clf, save_path)
        logger.info("Model saved to %s", save_path)

    except Exception:
        logger = logging.getLogger(__name__)
        logger.exception("An error occurred during training.")
        sys.exit(1)

if __name__ == "__main__":
    main()