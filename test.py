import re
import nltk
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from nltk.stem import PorterStemmer
from util import *

delimiters = r'[.?!]'

def naive(text):
    """
    Naively segments a paragraph into sentences using regex delimiters.
    """
    if isinstance(text, str):
        segments = re.split(delimiters, text)
        segmentedText = [s.strip() for s in segments if s.strip()]
    else:
        print("Input text must be a string.")
        segmentedText = []
    return segmentedText

def punkt(text):
    """
    Segments text using NLTK's PunktSentenceTokenizer.
    """
    if isinstance(text, str):
        tokenizer = PunktSentenceTokenizer()
        segmentedText = tokenizer.tokenize(text)
        return segmentedText 
    else:
        print("Input must be a string.")
        return []

def reduce(text):
    """
    Applies Porter stemming to each word in a list of tokenized sentences.
    """
    if isinstance(text, list):
        ps = PorterStemmer()
        reducedText = [
            [ps.stem(word) for word in sentence] if isinstance(sentence, list) else sentence
            for sentence in text
        ]
    else:
        print("Input must be a list of lists of words.")
        reducedText = None
        
    return reducedText

def pennTreeBank(text):
    """
    Tokenizes each sentence using the Treebank tokenizer.
    """
    tokenizedText = []
    if isinstance(text, list):
        tokenizer = TreebankWordTokenizer()
        for sentence in text:
            if isinstance(sentence, str):
                segment_tokens = tokenizer.tokenize(sentence)
                tokenizedText.append(segment_tokens)
            else:
                tokenizedText.append([])
    else:
        print("Input must be a list of strings.")
                
    return tokenizedText

# Example usage
output = pennTreeBank(["The well-known actor starred in a record-breaking award-winning performance"])
print(output)
