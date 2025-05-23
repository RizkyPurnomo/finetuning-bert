# Fine-Tuning BERT for Classification Task
This project is my journal where I documented my journey to find the best performing BERT model through thoughtful data preprocessing and fine-tuning.
For preprocessing, I applied combination of text cleaning, stopword removal, and lemmatization (I tried all the combination!). The fine-tuning process involved several key steps: choosing a pre-trained model that's tailored to specific domains, tuning hyperparameters, and experimenting with different features and classifiers to find the best setup.
The tasks I was working on include sentiment analysis [(SST-2)](https://huggingface.co/datasets/stanfordnlp/sst2), topic classification ([AG News](https://huggingface.co/datasets/fancyzhx/ag_news)), and spam detection ([Enron Spam](https://huggingface.co/datasets/SetFit/enron_spam)) using datasets from Hugging Face.

---

## Libraries used:
- `scikit-learn` (for classifiers, evaluation metrics, and parameter tuning)
- `nltk` (for text preprocessing like tokenization and lemmatization)
- `torch` and `transformers` (to load and fine-tune BERT models)
- `datasets` (to easily access benchmark datasets)
- `wandb` (for tracking experiments and logging results)

## Results

### Accuracy of Preprocessing Methods
| Preprocessing             | SST-2  | AG News | Enron Spam |
|:---------------------------|:--------:|:---------:|:-------------:|
| **Do Nothing**            | **0.869** | 0.837   | 0.977       |
| **Cleaning (C)**          | 0.860  | 0.840   | **0.982**   |
| Lemmatization (L)         | 0.860  | 0.840   | 0.978       |
| Stopword Removal (SR)     | 0.830  | 0.847   | 0.976       |
| (C) → (L)                 | 0.850  | 0.841   | 0.981       |
| (C) → (SR)                | 0.813  | 0.837   | 0.977       |
| (L) → (C)                 | 0.848  | 0.845   | 0.981       |
| **(L) → (SR)**            | 0.822  | **0.851** | 0.977     |
| (SR) → (C)                | 0.812  | 0.836   | 0.977       |
| (SR) → (L)                | 0.821  | 0.839   | 0.978       |
| (C) → (L) → (SR)          | 0.805  | 0.843   | 0.974       |
| (C) → (SR) → (L)          | 0.812  | 0.842   | 0.975       |
| (L) → (C) → (SR)          | 0.806  | 0.838   | 0.975       |
| (L) → (SR) → (C)          | 0.807  | 0.836   | 0.975       |
| (SR) → (C) → (L)          | 0.808  | 0.839   | 0.975       |
| (SR) → (L) → (C)          | 0.807  | 0.842   | 0.975       |

For shorter text, it's better to do nothing or minimalize preprocessing. 

### Accuracy of Domain-Specific Models
| Model                        | SST-2  | AG News | Enron Spam |
|:-----------------------------|:--------:|:---------:|:-------------:|
| BERT-base                   | 0.868  | 0.854   | 0.981       |
| **BERT-large**              | 0.892  | 0.852   | **0.984**   |
| BERT-tiny-enron-spam        | 0.726  | 0.800   | 0.977       |
| **BERT-base-ag-news**       | 0.824  | **0.963** | 0.976     |
| **BERT-base-sst-2**         | **0.983** | 0.843 | 0.978     |

Fine-tuned small/base-sized models competed with large-sized models.

### Accuracy of Hyperparameter Tuning
| Batch Size | Learning Rate | Epochs | SST-2       | AG News     | Enron Spam |
|:------------|:----------------|:--------|:-------------:|:-------------:|:------------:|
| 16         | 5e-5           | 2      | 0.968       | 0.955       | 0.974      |
|            |                | 3      | 0.969       | 0.957       | 0.985      |
|            |                | 4      | 0.966       | 0.956       | 0.987      |
|            | 3e-5           | 2      | **0.979**   | 0.965       | 0.985      |
|            |                | 3      | 0.976       | 0.960       | 0.988      |
|            |                | 4      | **0.979**   | 0.965       | 0.990      |
|            | 2e-5           | 2      | **0.979**   | 0.970       | 0.990      |
|            |                | 3      | **0.979**   | 0.973       | 0.990      |
|            |                | 4      | **0.979**   | 0.972       | 0.988      |
| 32         | 5e-5           | 2      | 0.977       | 0.961       | 0.984      |
|            |                | 3      | 0.975       | 0.961       | **0.991**  |
|            |                | 4      | 0.976       | 0.963       | 0.989      |
|            | 3e-5           | 2      | **0.979**   | 0.973       | 0.979      |
|            |                | 3      | 0.977       | 0.971       | 0.988      |
|            |                | 4      | 0.977       | 0.973       | **0.991**  |
|            | 2e-5           | 2      | **0.979**   | 0.973       | 0.990      |
|            |                | 3      | **0.979**   | 0.973       | 0.987      |
|            |                | 4      | **0.979**   | **0.976**   | 0.987      |

Learning rate 2e-5 and batch size 32 worked better.

### Accuracy of Feature Based Approach
| Feature                        |   SST-2   | AG News  | Enron Spam |
|:------------------------------|:---------:|:--------:|:----------:|
| Embeddings Layer              |   0.625   |  0.819   |   0.940    |
| **Last Hidden Layer**         | **0.979** | **0.965**| **0.989**  |
| Second-to-Last Hidden Layer   |   0.978   |  0.962   |   0.987    |
| Sum of All Hidden Layers      |   0.975   |  0.959   |   0.982    |
| Sum of Last 4 Hidden Layers   |   0.977   |  0.962   |   0.985    |
| Concat of Last 4 Hidden Layers|   0.977   |  0.961   |   0.985    |

The last hidden layer fitted well for classification task. 

### Accuracy of Classifiers
| Classifier                     |  SST-2  | AG News | Enron Spam |
|:-------------------------------|:-------:|:-------:|:----------:|
| **BERT Classification Head**     | **0.979** |  0.976  |  **0.990**  |
| Multilayer Perceptron (Sklearn)|  0.978  |  0.974  |   0.977    |
| Naive Bayes                    |  0.977  | **0.977** |   0.980    |
| Support Vector Machine         |  0.978  |  0.976  |   0.980    |
| Random Forest                  |  0.978  |  0.975  |   0.980    |

Machine learning classifiers competed with BERT's transformers built-in classification head
