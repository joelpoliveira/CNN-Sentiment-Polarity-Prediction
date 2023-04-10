import string
import nltk
from tqdm import tqdm
from typing import NewType

Word = NewType("Word", str)
Sentence = NewType("Sentence", list[Word])
Paragraph = NewType("Paragraph", list[Sentence])
Document = NewType("Document", list[Paragraph])


def __get_words(sentence: Sentence) -> Word:
    """
    Given a sentence, parses it's tokens, removing punctuation, stopwords and small words;
    It yields each word, one at a time.
    """
    stopwords = set(map(str.lower, nltk.corpus.stopwords.words("english")))
    punctuation = set(string.punctuation)
    for word in nltk.tokenize.wordpunct_tokenize(sentence):
        word = word.lower()
        if (word.isalnum()) \
        and (word not in punctuation) \
        and (word not in stopwords) \
        and (len(word) > 1):
            yield word 

            
def process_documents(documents: Document) -> dict:
    """
    Given a list of paragraphs, iterates over it's sentences. 
    Every time a new word is found, it is added to the dictionary of words with a unique integer reference.
    """
    all_words = {}
    sentences = []
    i=1
    
    for doc in tqdm(documents):
        
        for sentence in nltk.sent_tokenize(doc):
            current_sentence = []
            
            for word in __get_words(sentence):
                
                if word not in all_words:
                    all_words[ word ] = i
                    i+=1
                current_sentence.append(word)
                
            sentences.append(current_sentence)
            
    return all_words, sentences