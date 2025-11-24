# This script allows to run specific tasks of the full pipeline
# The tasks you want to run are controlled by flags (see below)

from constants import *

flags = {
    "prepare_noise": False,
    "extract_features": False,
    "view_features": False,
    "extract_labels": False,
    "view_labels": False,
    "train": True,
    "training_metrics": False,
    "test_model": False
}

if __name__ == "__main__":
    if flags["prepare_noise"]:
        from audio.noise import add_noise

        print("Preparing noisy data...")
        add_noise(
            speech_dir='data/test/speech/',
            noise_type='babble',
            snr_db=5,
            output_dir='data/test/noisy/'
        )
        print("Noisy data prepared.")

    if flags["extract_features"]:
        from data.features import extract_features
        
        print("Extracting features...")
        extract_features(
            input_dir='data/test/noisy/',
            output_dir='data/test/features/'
        )
        print("Feature extraction completed.")

    if flags["view_features"]:
        from util.view import plot_feature
        
        print("Visualizing a feature...")
        plot_feature(dir='data/train/features/', compare_speech=True)
        print("Feature visualization done.")

    if flags["extract_labels"]:
        from data.labels import extract_labels
        
        print("Extracting labels...")
        extract_labels(
            speech_dir='data/train/speech/',
            noisy_dir='data/train/noisy/',
            noise_dir='data/noise/',
            output_dir='data/train/labels/',
            mode='spectro' # ibm, irm or spectro
        )
        print("Label extraction completed.")

    if flags["view_labels"]:
        from util.view import plot_feature
        
        print("Visualizing a label...")
        plot_feature(dir='data/train/labels/')
        print("Label visualization done.")

    if flags["train"]:
        from cnn.train import train_cnn_model

        print("Training CNN model...")
        train_cnn_model(
            log_dir='./logs/cnn/model_8/',
            features_dir='data/train/features/',
            labels_dir='data/train/labels/',
            mode='spectro' # ibm, irm or spectro
        )
        print("CNN model training completed.")

    if flags["training_metrics"]:
        from util.metrics import analyse_training_run

        print("Analysing training metrics...")
        analyse_training_run(log_dir='./logs/cnn/model_7/')
        print("Training metrics analysis done.")

    if flags["test_model"]:
        from cnn.test import test_cnn_model

        print("Testing CNN model...")
        test_cnn_model(
            model_path='./logs/cnn/model_7/models/cnn_final.pt',
            test_samples_path='data/test/noisy/',
            output_dir='data/test/enhanced/',
            mode='spectro',  # ibm, irm or spectro
            sample_test=True
        )
        print("CNN model testing completed.")
