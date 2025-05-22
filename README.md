# Fine-Tuning BERT for Classification Task
This project is my journal where I document my journey to find the best performing BERT model through thoughtful data preprocessing and fine-tuning.
For preprocessing, I applied combination of text cleaning, stopword removal, and lemmatization (I tried all the combination!). The fine-tuning process involves several key steps: choosing a pre-trained model that's tailored to specific domains, tuning hyperparameters, and experimenting with different features and classifiers to find the best setup.
The tasks I’m working on include news topic classification, sentiment analysis, and spam detection—using datasets from Hugging Face.



This project is my personal diary to get the best performing BERT through data preprocessing and fine-tuning. The preprocessing used in this study is cleaning, stopword removal, and lemmatization. In doing fine-tuning in the classification task, several stages are carried out such as selecting a pre-trained model that is specific to a certain domain, hyperparameter tuning, finding the best feature and classifier. The tasks chosen are news topic classification, sentiment analysis, and spam detection in the Hugging Face dataset.

---

## Libraries used:
- scikit-learn (for classifiers, evaluation metrics, and parameter tuning)
- nltk (for text preprocessing like tokenization and lemmatization)
- transformers (to load and fine-tune BERT models)
- datasets (to easily access benchmark datasets)
- wandb (for tracking experiments and logging results)