import string
import nltk
import numpy as np
from tqdm.notebook import tqdm
from typing import NewType, Sequence
from googletrans import Translator

Word = NewType("Word", str)
WordIndex = NewType("WordIndex", int)

Document = NewType("Sentence", list[Word])
IndexedDocument = NewType("IndexSentences", list[WordIndex])

Corpus = NewType("Corpus", list[Document])
IndexedCorpus = NewType("IndexedCorpus", list[IndexedDocument])
Vocabulary = NewType("Vocabulary", dict[Word, int])

TRANSLATOR = Translator()
    
def __translate(document: str) -> str:
    """
    Uses Google Translator API to translate {document} to english
    """
    return TRANSLATOR.translate(document).text

        
def translate_to_english(documents: list[str])->list[str]:
    """
    Translates the documents in {documents} to english
    """
    translated = []
    for doc in documents:
        translated.append(__translate(doc))
    return translated


def batch_translate_to_english(documents: list[str], batch_size=10) -> list[str]:
    """
    Translates the documents in {documents} to english, in batches
    """
    translated = []
    for i in range(0, len(documents), batch_size):
        batch = '\n'.join(documents[i:i+batch_size])
        translation = __translate(batch)
        translated.extend(
            list(map(lambda document: document+" \n", translation.split("\n")))
        )
    return translated


def __remove_punctuation(document: str)->str:
    """
    Removes punctuation and special characters from {document}
    """
    n = len(string.punctuation)
    punct_trans = dict(
        zip(string.punctuation, [' ']*n)
    )
    return document.translate(str.maketrans(punct_trans))

def __get_words(document: str) -> Document:
    """
    Given a document, parses it's tokens, removing punctuation, stopwords and small words;
    It yields each word, one at a time.
    """
    stopwords = set(map(str.lower, nltk.corpus.stopwords.words("english")))
    for word in nltk.tokenize.wordpunct_tokenize(document):
        word = word.lower()
        if (word.isalpha()) \
        and (word not in stopwords) \
        and (len(word) > 1):
            yield word 


def process_documents(documents: list[str], return_vocab=True) -> Corpus: 
    """
    Given a raw corpus of string documents, parses each document, tokenizing it. 
    Punctuation are filtered from each token. Stopwords and small words are also filtered.
    
    If {return_vocab} is set to "True" (default) a dictionary is built with the indexing of 
    each unique word in the corpus.
    """
    if return_vocab:
        all_words = {}
    
    new_docs = []
    i=1
    
    for doc in tqdm(documents):
        current_doc = []
        for sentence in nltk.sent_tokenize(doc):
            sentence = __remove_punctuation(sentence)
        
            for word in __get_words(sentence):
                if return_vocab:
                    if word not in all_words:
                        all_words[ word ] = i
                        i+=1
                    
                current_doc.append(word)
        new_docs.append(current_doc)
    
    if return_vocab:
        return all_words, new_docs
    return new_docs


def add_unknown_words(words, word2vec, dev=.25, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector. 
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    
    vecs = [np.random.uniform(-dev, dev,k) for _ in range(len(words))]
    word2vec.add_vectors(words, vecs)


def map_to_index(documents: Corpus, vocab: Vocabulary, UNKOWN_TOKEN: Word = "[UNK]") -> IndexedCorpus:
    """
    Parses each token in {documents} replacing it's value with the respective index from {vocab}. 
    If the token is not present in the vocabulary it will be replaced with the index of the {UNKOWN_TOKEN}.
    """
    index_docs = documents.copy()
    for i, doc in tqdm(enumerate(documents)):
        for j, word in enumerate(doc):
            if word in vocab:
                index_docs[i][j] = vocab[word]
            else:
                index_docs[i][j] = vocab[UNKOWN_TOKEN]
                
    return index_docs


def get_max_sequence_length(documents: IndexedCorpus) -> int:
    """Calculates the maximum token length present in {documents}"""
    return max(
        [len(d) for d in documents]
    )


def empty_padding(document: IndexedDocument, max_len:int, pad_index:int) -> IndexedDocument:
    """
    Sets the length of {document} to {max_len}, either by padding or by truncating.
    """
    pad_len = max_len - len(document)
    if pad_len < 0:
        return document[:max_len]
    return document + [pad_index] * pad_len


def pad_documents(documents: IndexedCorpus, max_len:int, pad_index:int) -> IndexedCorpus:
    """
    Sets {documents} to a tabular format of shape ( n_documents, max_len )
    """
    return np.array(
        list(map(lambda s: empty_padding(s, max_len, pad_index), tqdm(documents)))
    )