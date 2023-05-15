import string
import nltk
import numpy as np
from tqdm.notebook import tqdm
from typing import NewType, Sequence
from deep_translator import GoogleTranslator
from googletrans import Translator
from multiprocessing.pool import ThreadPool as Pool

Word = NewType("Word", str)
WordIndex = NewType("WordIndex", int)

Sentence = NewType("Sentence", list[Word])
IndexSentence = NewType("IndexSentences", list[WordIndex])

Paragraph = NewType("Paragraph", list[Sentence])

Document = NewType("Document", list[Paragraph])
Vocabulary = NewType("Vocabulary", dict[Word, int])
TRANSLATOR = Translator()
    
def __translate(sentence: Sentence) -> Sentence:
    return TRANSLATOR.translate(sentence).text

        
    
def translate_to_english(documents: list[Document]):
    translated = []
    for doc in documents:
        translated.append(__translate(doc))
    return translated


def batch_translate_to_english(documents: list[Document], batch_size=10):
    translated = []
    for i in range(0, len(documents), batch_size):
        batch = '\n'.join(documents[i:i+batch_size])
        translation = __translate(batch)
        translated.extend(
            list(map(lambda sentence: sentence+" \n", translation.split("\n")))
        )
    return translated


def __remove_punctuation(sentence: Sentence)->Sentence:
    n = len(string.punctuation)
    punct_trans = dict(
        zip(string.punctuation, [' ']*n)
    )
    return sentence.translate(str.maketrans(punct_trans))

def __get_words(sentence: Sentence) -> Sequence[Word]:
    """
    Given a sentence, parses it's tokens, removing punctuation, stopwords and small words;
    It yields each word, one at a time.
    """
    stopwords = set(map(str.lower, nltk.corpus.stopwords.words("english")))
    for word in nltk.tokenize.wordpunct_tokenize(sentence):
        word = word.lower()
        if (word.isalpha()) \
        and (word not in stopwords) \
        and (len(word) > 1):
            yield word 


def process_documents(documents: list[Document], return_vocab=True):       
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


def process_sentences(sentences: list[Sentence]) -> dict:
    """
    Given a list of paragraphs, iterates over it's sentences. 
    Every time a new word is found, it is added to the dictionary of words with a unique integer reference.
    """
    all_words = {}
    new_sentences = []
    i=1
    
    for sentence in tqdm(sentences):
            current_sentence = []
            #sentence = __translate(sentence)
            sentence = __remove_punctuation(sentence)
            
            for word in __get_words(sentence):
                
                if word not in all_words:
                    all_words[ word ] = i
                    i+=1
                current_sentence.append(word)
                
            new_sentences.append(current_sentence)
            
    return all_words, new_sentences



def add_unknown_words(words, word2vec, dev=.25, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector. 
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    
    vecs = [np.random.uniform(-dev, dev,k) for _ in range(len(words))]
    word2vec.add_vectors(words, vecs)


def map_to_index(documents: list, vocab: Vocabulary, UNKOWN_TOKEN: Word = "[UNK]") -> list[IndexSentence]:
    index_docs = documents.copy()
    for i, doc in tqdm(enumerate(documents)):
        for j, word in enumerate(doc):
            if word in vocab:
                index_docs[i][j] = vocab[word]
            else:
                index_docs[i][j] = vocab[UNKOWN_TOKEN]
                
    return index_docs
    
    
def map_sentences(sentences: list[Sentence], vocab: Vocabulary, UNKOWN_TOKEN: Word = "[UNK]") -> list[Sentence]:
    """
    Given a list of sentences, iterates over it. 
    Maps each word to the corresponding index in the vocab if it exists.
    """
    mapped_sentences = []
    for sentence in tqdm(sentences):
        current_sentence = []
        #sentence = __translate(sentence)
        sentence = __remove_punctuation(sentence)
            
        for word in __get_words(sentence):
            if word in vocab:
                current_sentence.append( vocab[word] )
            else:
                current_sentence.append( vocab[UNKOWN_TOKEN] )
                
        if len(current_sentence)==0:
            print(sentence)
            break
        mapped_sentences.append(current_sentence)
            
    return mapped_sentences


def map_documents(documents: list[Document], vocab: Vocabulary, UNKOWN_TOKEN: Word = "[UNK]") -> list[Document]:
    """
    Given a list of sentences, iterates over it. 
    Maps each word to the corresponding index in the vocab if it exists.
    """
    mapped_docs = []
    for doc in tqdm(documents):
        current_doc = []
        for sentence in nltk.sent_tokenize(doc):
            #sentence = __translate(sentence)
            sentence = __remove_punctuation(sentence)
            
            for word in __get_words(sentence):
                if word in vocab:
                    current_doc.append( vocab[word] )
                else:
                    current_doc.append( vocab[UNKOWN_TOKEN] )

        mapped_docs.append(current_doc)
            
    return mapped_docs




def get_max_sequence_length(sentences: IndexSentence) -> int:
    """Returns the """
    return max(
        [len(s) for s in sentences]
    )


def empty_padding(sentence: IndexSentence, max_len:int, pad_index:int) -> IndexSentence:
    pad_len = max_len - len(sentence)
    if pad_len < 0:
        return sentence[:max_len]
    return sentence + [pad_index] * pad_len


def pad_sentences(sentences: list[IndexSentence], max_len:int, pad_index:int) -> list[IndexSentence]:
    return np.array(
        list(map(lambda s: empty_padding(s, max_len, pad_index), sentences))
    )