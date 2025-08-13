from util import *
from nltk.tokenize import TreebankWordTokenizer
import re

class Tokenization():
    
    def naive(self, text):
        """
        Tokenizes each sentence using basic regex-based splitting.
        Removes punctuation tokens from the result.
        """
        tokenizedText = []
        if isinstance(text, list):
            for sentence in text:
                if isinstance(sentence, str):
                    # Split using pre-defined separators
                    segment_tokens = re.split(text_separators, sentence)
                    # Filter out punctuation
                    cleaned_tokens = [token for token in segment_tokens if token not in punctuations and token.strip()]
                    tokenizedText.append(cleaned_tokens)
        else:
            print("Invalid input: Expected a list of strings.")
        
        return tokenizedText 
    
    def pennTreeBank(self, text):
        """
        Tokenizes each sentence using the Penn Treebank tokenizer.
        """
        tokenizedText = []
        if isinstance(text, list):
            tokenizer = TreebankWordTokenizer()
            for sentence in text:
                if isinstance(sentence, str):
                    segment_tokens = tokenizer.tokenize(sentence)
                    tokenizedText.append(segment_tokens)
        else:
            print("Invalid input: Expected a list of strings.")
                    
        return tokenizedText
