import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
import helpers
import bert_model

# Pastikan resource NLTK sudah diunduh
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

# Inisialisasi lemmatizer dan daftar stopwords
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Membersihkan teks dari karakter non-alfanumerik dan merubahnya ke huruf kecil.
    Args:
        text (str): Teks input.
    Returns:
        str: Teks yang telah dibersihkan.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Hapus karakter non-alfanumerik
    text = text.lower()  # Ubah ke huruf kecil
    return text


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

def lemmatize_text(text):
    """
    Melakukan lemmatization pada setiap kata dalam teks dengan mempertimbangkan POS tag.
    Args:
        text (str): Teks input.
    Returns:
        str: Teks yang telah dilemmatize.
    """
    tokens = nltk.word_tokenize(text)  # Tokenisasi
    pos_tags = nltk.pos_tag(tokens)  # POS tagging
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags
    ]  # Lemmatize dengan POS
    return ' '.join(lemmatized)


def stem_text(text):
    """
    Melakukan lemmatization pada setiap kata dalam teks dengan mempertimbangkan POS tag.
    Args:
        text (str): Teks input.
    Returns:
        str: Teks yang telah dilemmatize.
    """
    tokens = nltk.word_tokenize(text)  # Tokenisasi
    filtered = [stemmer.stem(word) for word in tokens]  # Stem
    return ' '.join(filtered)


def remove_stopwords(text):
    """
    Menghapus stopwords dari teks.
    Args:
        text (str): Teks input.
    Returns:
        str: Teks tanpa stopwords.
    """
    tokens = nltk.word_tokenize(text)  # Tokenisasi
    filtered = [word for word in tokens if word not in stop_words]  # Hapus stopwords
    return ' '.join(filtered)


def apply_preprocessing(text, methods):
    """
    Mengaplikasikan kombinasi metode preprocessing pada teks.
    Args:
        text (str): Teks input.
        methods (list): Daftar metode preprocessing dalam urutan yang diinginkan. 
                        Pilihan: ['C', 'L', 'SR'].
    Returns:
        str: Teks yang telah diproses.
    """
    for method in methods:
        if method == 'C':
            text = clean_text(text)
        elif method == 'L':
            text = lemmatize_text(text)
        elif method == 'SR':
            text = remove_stopwords(text)
    return text


def preprocess_pipeline(texts, method_sequence):
    """
    Melakukan preprocessing pada seluruh dataset.
    Args:
        texts (list): List teks.
        method_sequence (list): Urutan metode preprocessing.
    Returns:
        list: List teks yang telah diproses.
    """
    return [apply_preprocessing(text, method_sequence) for text in texts]


def evaluate_preprocessing_methods(dataset_name):
    """
    Mengevaluasi kombinasi metode preprocessing pada dataset tertentu.
    Args:
        dataset_name (str): Nama dataset (sst2, ag_news, enron_spam).
    """
    method_combinations = [
        [],  # X (Do Nothing)
        ['C'], ['L'], ['SR'],
        ['C', 'L'], ['C', 'SR'], ['L', 'C'], ['L', 'SR'],
        ['SR', 'C'], ['SR', 'L'],
        ['C', 'L', 'SR'], ['C', 'SR', 'L'], ['L', 'C', 'SR'],
        ['L', 'SR', 'C'], ['SR', 'C', 'L'], ['SR', 'L', 'C']
    ]

    results = {}
    for methods in method_combinations:
        method_name = " -> ".join(methods) if methods else "Do Nothing (X)"
        print('Metode:', method_name)
        result = bert_model.evaluate_model(
            dataset_name=dataset_name,
            preprocess_method=methods
        )

        # Save results
        results[method_name] = result
        print(f"Methods: {method_name} | Metrics: {result}")
        print('results =', results)

    # Save results to file
    output_file = f"results_preprocessing_{dataset_name}.json"
    helpers.save_results_to_file(results, output_file)
    print(f"Results saved to {output_file}")

    # Find the best method based on accuracy
    best_method = max(results, key=lambda m: results[m])
    print(
        f"Best preprocessing method for {dataset_name}: {best_method} with accuracy {results[best_method]}"
    )

    return best_method
