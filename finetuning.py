import helpers
import bert_model
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import wandb
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterSampler


# Fine-Tuning Step 1: Domain-Specific Models
def fine_tuning_step_1(dataset_name, preprocessing_method):
    """
    Fine-Tuning Step 1: Test several domain-specific BERT models.

    Parameters:
    - dataset (Dataset): The dataset to be used for fine-tuning.

    Returns:
    - best_model (PreTrainedModel): The best domain-specific model after fine-tuning.
    """
    # Available models
    models = [
        'google-bert/bert-base-uncased',
        'google-bert/bert-large-uncased',
        'mrm8488/bert-tiny-finetuned-enron-spam-detection',
        'textattack/bert-base-uncased-ag-news',
        'textattack/bert-base-uncased-SST-2'
    ]
    
    # Initialize best model tracking
    results = {}
    for model_name in models:
        print('Model:', model_name)
        result = bert_model.evaluate_model(
            dataset_name=dataset_name,
            preprocess_method=preprocessing_method,
            model_name=model_name
        )

        # Save results
        results[model_name] = result
        print(f"Model: {model_name} | Metrics: {result}")
        print('results =', results)

    # Save results to file
    output_file = f"results_model_domain_{dataset_name}.json"
    helpers.save_results_to_file(results, output_file)
    print(f"Results saved to {output_file}")

    # Find the best method based on accuracy
    best_model_name = max(results, key=lambda m: results[m])
    print(
        f"Best BERT model for {dataset_name}: {best_model_name} with accuracy {results[best_model_name]}"
    )
    return best_model_name


# Fine-Tuning Step 2: Hyperparameter Tuning
def fine_tuning_step_2(dataset_name, preprocessing_method, model_name):
    """
    Fine-Tuning Step 2: Hyperparameter tuning for batch size, learning rate, and epochs.

    Parameters:
    - model (PreTrainedModel): The model to be fine-tuned.
    - dataset (Dataset): The dataset used for training.

    Returns:
    - best_model (PreTrainedModel): The best model after hyperparameter tuning.
    - best_params (dict): Best hyperparameters.
    """
    best_params = None
    best_score = 0

    batch_sizes = [16, 32]
    learning_rates = [5e-5, 3e-5, 2e-5]
    epochs = [4]

    results = {}
    for batch_size in batch_sizes:
        for lr in learning_rates:
            for epoch in epochs:
                print(f"Testing batch_size={batch_size}, learning_rate={lr}, epochs={epoch}")
                hyperparameter_config_str = str([batch_size, lr, epoch])
                result = bert_model.evaluate_model(
                    dataset_name=dataset_name,
                    preprocess_method=preprocessing_method,
                    model_name=model_name,
                    batch_size=batch_size,
                    learning_rate=lr,
                    epochs=epoch
                )

                # Save results
                results[hyperparameter_config_str] = result
                print(f"Hyperparameter: {hyperparameter_config_str} | Metrics: {result}")
                print('results =', results)

    # Save results to file
    output_file = f"results_model_domain_{dataset_name}.json"
    helpers.save_results_to_file(results, output_file)
    print(f"Results saved to {output_file}")

    # Find the best method based on accuracy
    best_hyperparameter_config_str = max(results, key=lambda m: results[m])
    print(
        f"Best hyperparameter combination model for {dataset_name}: {best_hyperparameter_config_str} "
        f"with accuracy {results[best_hyperparameter_config_str]}"
    )

    best_hyperparameter_config = (best_hyperparameter_config_str.replace(",", "").replace("[", "").replace("]", "").split())
    best_batch_size = int(best_hyperparameter_config[0])
    best_learning_rate = float(best_hyperparameter_config[1])
    best_epoch = int(best_hyperparameter_config[2])

    return best_batch_size, best_learning_rate, best_epoch


def fine_tuning_step_3(dataset_name, preprocessing_method, model_name, batch_size, learning_rate, epochs):
    """
    Fine-Tuning Step 3: Feature-Based Approach for extracting the best features.
    This step will test various feature extraction methods such as embeddings and layers.

    Parameters:
    - model (PreTrainedModel): The best model from Step 2.
    - dataset (Dataset): The dataset used for feature extraction.

    Returns:
    - best_model (PreTrainedModel): The best model after feature extraction.
    """
    print("Starting Feature-Based Approach...")

    # Feature extraction methods: embeddings, last layer, second-to-last layer, etc.
    feature_extraction_methods = [
        "embeddings", "last_layer", "second_to_last_layer", 
        "sum_all_layers", "sum_last_four_layers", "concat_last_four_layers"
    ]
    
    results = {}
    
    for feature in feature_extraction_methods:
        print(f"Feature Extraction Method: {feature}")

        result = bert_model.evaluate_model(
            dataset_name=dataset_name,
            preprocess_method=preprocessing_method,
            model_name=model_name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            feature_extraction_method=feature
        )

        # Save results
        results[feature] = result
        print(f"Feature: {feature} | Metrics: {result}")

    # Save results to file
    output_file = f"results_feature_{dataset_name}.json"
    helpers.save_results_to_file(results, output_file)
    print(f"Results saved to {output_file}")

    # Find the best method based on accuracy
    best_feature = max(results, key=lambda m: results[m])
    print(
        f"Best feature model for {dataset_name}: {best_feature} "
        f"with accuracy {results[best_feature]}"
    )

    return best_feature


# Fine-Tuning Step 4: Classifier Testing
def fine_tuning_step_4(dataset_name, preprocessing_method, model_name, batch_size, learning_rate, epochs, feature,
                       **kwargs,
                       # hidden_layer_sizes=(768,), activation='relu', solver='adam',
                       # alpha=1.0,
                       # C=1.0, kernel='rbf', gamma='scale',
                       # n_estimators=100, max_depth=None, min_samples_split=2
                       ):
    """
    Fine-Tuning Step 4: Test different classifiers (Naive Bayes, SVM, MLP, Random Forest).

    Parameters:
    - model: The best model from previous steps.
    - dataset: The dataset used for training and testing.
    - params: Hyperparameters for the classifier.

    Returns:
    - final_model: The best model after classifier testing.
    """
    print("Searching for the best classifier..")

    param_grids = {
        "Naive Bayes": {
            "var_smoothing": [1e-9],
        },
        "SVM": {
            "C": np.logspace(-2, 1, 10),
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
        "Random Forest": {
            "n_estimators": np.arange(50, 201, 50),
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": np.arange(2, 11, 2),
        },
        "MLP": {
            "hidden_layer_sizes": [(256,), (512,), (768,), (1024,)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
            "alpha": np.logspace(-4, -1, 10),
        },
        
    }

    results = {}

    for classifier_name, param_grid in param_grids.items():
        print(f"Starting random search for classifier: {classifier_name}")

        # Generate random samples of hyperparameters
        param_samples = list(ParameterSampler(param_grid, n_iter=5, random_state=42))

        best_score = 0
        best_params = None

        # Iterate over sampled parameter combinations
        for params in param_samples:
            print(f"Testing parameters: {params}")

            # Evaluate model with current parameters
            result = bert_model.evaluate_model(
                dataset_name=dataset_name,
                preprocess_method=preprocessing_method,
                model_name=model_name,
                batch_size=batch_size,
                learning_rate=learning_rate,
                epochs=epochs,
                feature_extraction_method=feature,
                classifier_name=classifier_name,
                **params,  # Pass sampled parameters
            )

            # Update best parameters if current score is better
            if result > best_score:
                best_score = result
                best_params = params

        # Save results for the current classifier
        results[classifier_name] = {
            "best_params": best_params,
            "best_score": best_score,
        }
        print(f"Best parameters for {classifier_name}: {best_params}")
        print(f"Best score: {best_score}")

    # Save results to file
    output_file = f"results_classifier_{dataset_name}.json"
    helpers.save_results_to_file(results, output_file)
    print(f"Results saved to {output_file}")

    best_classifier_name = max(results, key=lambda m: results[m]["best_score"])
    best_classifier_values = results[best_classifier_name]

    print(f"Best Classifier: {best_classifier_name}")
    print(f"Best Parameters: {best_classifier_values['best_params']}")
    print(f"Best Score: {best_classifier_values['best_score']}")

    return best_classifier_name, best_classifier_values['best_params'], best_classifier_values['best_score']


# Metrics computation function
def compute_metrics(p):
    """
    Compute the metrics such as accuracy, precision, recall, F1-score.
    """
    preds, labels = p
    preds = np.argmax(preds, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
