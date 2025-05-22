import os
import helpers
import preprocessing
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm


def evaluate_model(dataset_name='sst2',
                   preprocess_method=['C', 'L', 'SR'],
                   model_name='google-bert/bert-base-uncased',
                   batch_size=16, learning_rate=2e-5, epochs=1,
                   feature_extraction_method='last_layer',
                   classifier_name='MLP',
                   **kwargs):

    train_dataset, eval_dataset = helpers.load_and_prepare_dataset(dataset_name)
    # print('Before Preprocess:', train_dataset[0])
    train_dataset = train_dataset.map(
        lambda example: {'text': preprocessing.apply_preprocessing(example['text'], preprocess_method)}
    )
    eval_dataset = eval_dataset.map(
        lambda example: {'text': preprocessing.apply_preprocessing(example['text'], preprocess_method)}
    )
    # print('After Preprocess:', train_dataset[0])

    tokenizer = BertTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_tensors='pt', padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_dataset.map(
        lambda examples: {**tokenize_function(examples), "labels": examples["label"]},
        batched=True
    )
    tokenized_eval = eval_dataset.map(
        lambda examples: {**tokenize_function(examples), "labels": examples["label"]},
        batched=True
    )
    # print('After Tokenizing:', tokenized_train[0])

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(train_dataset.unique("label")),
        output_hidden_states=True,
        ignore_mismatched_sizes=True
    )

    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    model.to(device)
    print('model moved to', device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        report_to="wandb",  # Log to wandb
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=helpers.compute_metrics
    )

    # Train the model
    # print("Starting training...")
    trainer.train()

    # EXTRACT THE FEATURE using extract_methods() from the model
    # print("Extracting features using method:", feature_extraction_method)
    train_features, train_labels = extract_features(
        model=model,
        dataset=tokenized_train,
        feature_method=feature_extraction_method
    )
    eval_features, eval_labels = extract_features(
        model=model,
        dataset=tokenized_eval,
        feature_method=feature_extraction_method
    )

    # FEED THE EXTRACTED FEATURE TO THE CLASSIFIER defined in classifier() method
    # Step 8: Classifier
    # print("Evaluating classifier:", classifier_name)
    classifier = get_classifier(classifier_name)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(eval_features)

    # Evaluate model
    final_metrics = helpers.compute_final_metrics((predictions, eval_labels))
    print("Evaluation results:", final_metrics)

    return final_metrics['accuracy']


def extract_features(model, dataset, feature_method):
    """
    Extract features from a dataset using a BERT model and specified feature extraction method.

    Args:
        model (BertModel): Pretrained BERT model.
        dataset (Dataset): Tokenized dataset.
        feature_method (str): Feature extraction method. Options:
            - 'last_layer': Use the last hidden layer.
            - 'second_to_last_layer': Use the second-to-last hidden layer.
            - 'sum_all_layers': Sum outputs from all layers.
            - 'sum_last_four_layers': Sum outputs from the last four layers.
            - 'concat_last_four_layers': Concatenate outputs from the last four layers.

    Returns:
        features (np.ndarray): Extracted features.
        labels (np.ndarray): Corresponding labels.
    """
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    features = []
    labels = []

    with torch.no_grad():
        for example in tqdm(dataset, desc="Extracting Features"):
            # Convert input to PyTorch tensors
            input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(device)

            # Forward pass through the model
            outputs = model(input_ids, attention_mask=attention_mask)
            hidden_states = outputs.hidden_states  # List of all layer outputs

            # Extract features based on the method
            if feature_method == 'last_layer':
                feature = hidden_states[-1].squeeze(0).mean(dim=0).cpu().numpy()
            elif feature_method == 'second_to_last_layer':
                feature = hidden_states[-2].squeeze(0).mean(dim=0).cpu().numpy()
            elif feature_method == 'sum_all_layers':
                feature = torch.stack(hidden_states).sum(dim=0).squeeze(0).mean(dim=0).cpu().numpy()
            elif feature_method == 'sum_last_four_layers':
                feature = torch.stack(hidden_states[-4:]).sum(dim=0).squeeze(0).mean(dim=0).cpu().numpy()
            elif feature_method == 'concat_last_four_layers':
                feature = torch.cat(hidden_states[-4:], dim=-1).squeeze(0).mean(dim=0).cpu().numpy()
            elif feature_method == 'embeddings':
                feature = hidden_states[0].squeeze(0).mean(dim=0).cpu().numpy()
            else:
                raise ValueError(f"Unsupported feature extraction method: {feature_method}")

            features.append(feature)
            labels.append(example['label'])

    # Convert to NumPy arrays
    features = torch.tensor(features).numpy()
    labels = torch.tensor(labels).numpy()

    return features, labels


def get_classifier(classifier_name,
                   hidden_layer_sizes=(768,), activation='relu', solver='adam',
                   alpha=1.0,
                   C=1.0, kernel='rbf', gamma='scale',
                   n_estimators=100, max_depth=None, min_samples_split=2):
    if classifier_name == "MLP":
        return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver)
    elif classifier_name == "Naive Bayes":
        return GaussianNB()
    elif classifier_name == "SVM":
        return SVC(C=C, kernel=kernel, gamma=gamma)
    elif classifier_name == "Random Forest":
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)