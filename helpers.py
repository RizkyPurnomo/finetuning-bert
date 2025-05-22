import pandas as pd
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from datasets import load_dataset, concatenate_datasets
import wandb
import datasets



def get_dataset_names():
    """
    Return the list of supported datasets.
    
    Returns:
    - List[str]: Names of available datasets.
    """
    return ["sst2", "ag_news", "enron_spam"]


def load_and_prepare_dataset(dataset_name: str):
    """
    Load and prepare the dataset based on the dataset name.
    Ensures columns are standardized to 'text' and 'label'.

    Parameters:
    - dataset_name (str): The name of the dataset ('sst2', 'ag_news', or 'enron_spam').

    Returns:
    - pd.DataFrame: A Pandas DataFrame with 'text' and 'label' columns.
    """

    if dataset_name == "sst2":
        dataset = load_dataset("stanfordnlp/sst2", split='train[:3500]')
        dataset = dataset.rename_column("sentence", "text")
        dataset = dataset.remove_columns('idx')
        dataset = dataset.sort('label')

        train_0 = dataset.select(range(1000))  # First 1000 samples of label 0
        eval_0 = dataset.select(range(1000, 1500))  # Next 500 samples of label 0
        train_1 = dataset.select(range(2000, 3000))  # First 1000 samples of label 1
        eval_1 = dataset.select(range(3000, 3500))  # Next 500 samples of label 1

        train_dataset = concatenate_datasets([train_0, train_1]).shuffle(seed=42)
        eval_dataset = concatenate_datasets([eval_0, eval_1]).shuffle(seed=42)

    elif dataset_name == "ag_news":
        dataset = load_dataset("fancyzhx/ag_news", split='train[:7500]')
        dataset = dataset.sort('label')

        train_0 = dataset.select(range(1000))  # First 1000 samples of label 0
        eval_0 = dataset.select(range(1000, 1500))  # Next 500 samples of label 0
        train_1 = dataset.select(range(2000, 3000))  # First 1000 samples of label 1
        eval_1 = dataset.select(range(3000, 3500))  # Next 500 samples of label 1
        train_2 = dataset.select(range(4000, 5000))  # First 1000 samples of label 1
        eval_2 = dataset.select(range(5000, 5500))  # Next 500 samples of label 1
        train_3 = dataset.select(range(6000, 7000))  # First 1000 samples of label 1
        eval_3 = dataset.select(range(7000, 7500))  # Next 500 samples of label 1

        train_dataset = concatenate_datasets([train_0, train_1, train_2, train_3]).shuffle(seed=42)
        eval_dataset = concatenate_datasets([eval_0, eval_1, eval_2, eval_3]).shuffle(seed=42)

    elif dataset_name == "enron_spam":
        dataset = load_dataset("SetFit/enron_spam", split='train[:3500]')
        dataset = dataset.remove_columns(['message_id', 'label_text', 'subject', 'message', 'date'])
        dataset = dataset.sort('label')

        train_0 = dataset.select(range(1000))  # First 1000 samples of label 0
        eval_0 = dataset.select(range(1000, 1500))  # Next 500 samples of label 0
        train_1 = dataset.select(range(2000, 3000))  # First 1000 samples of label 1
        eval_1 = dataset.select(range(3000, 3500))  # Next 500 samples of label 1

        train_dataset = concatenate_datasets([train_0, train_1]).shuffle(seed=42)
        eval_dataset = concatenate_datasets([eval_0, eval_1]).shuffle(seed=42)

    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    dataset = None
    train_0 = None
    train_1 = None
    train_2 = None
    train_3 = None
    eval_0 = None
    eval_1 = None
    eval_2 = None
    eval_3 = None

    return train_dataset, eval_dataset


def save_results_to_file(results, file_path):
    """
    Menyimpan hasil ke file JSON untuk dokumentasi.
    Args:
        results (dict): Hasil eksperimen.
        file_path (str): Path file output.
    """
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)


def log_results_to_wandb(run_name, results):
    """
    Mencatat hasil ke wandb.ai untuk pelacakan eksperimen.
    Args:
        run_name (str): Nama eksperimen.
        results (dict): Hasil eksperimen.
    """
    wandb.init(project="bert-fine-tuning", name=run_name)
    wandb.log(results)
    wandb.finish()


def compute_metrics(p):
    """
    Compute the metrics such as accuracy, precision, recall, F1-score.
    """
    preds, labels = p
    preds = np.argmax(preds[0], axis=1)
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='macro'),
        'recall': recall_score(labels, preds, average='macro'),
        'f1_score': f1_score(labels, preds, average='macro'),
    }
    return metrics


def compute_final_metrics(p):
    """
    Compute the metrics such as accuracy, precision, recall, F1-score.
    """
    preds, labels = p
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='macro'),
        'recall': recall_score(labels, preds, average='macro'),
        'f1_score': f1_score(labels, preds, average='macro'),
    }
    return metrics


def save_model(model, file_path):
    """
    Menyimpan model yang telah dilatih.
    Args:
        model (transformers.PreTrainedModel): Model yang dilatih.
        file_path (str): Path file output.
    """
    torch.save(model.state_dict(), file_path)


def load_model(model_class, file_path):
    """
    Memuat model yang telah disimpan.
    Args:
        model_class: Kelas model (e.g., BertForSequenceClassification).
        file_path (str): Path file model yang disimpan.
    Returns:
        object: Model yang dimuat.
    """
    model = model_class.from_pretrained("bert-base-uncased")
    model.load_state_dict(torch.load(file_path))
    return model
