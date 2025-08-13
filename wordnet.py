from nltk.corpus import wordnet

def Synsets(word):
    """
    Fetch and display all synsets associated with a given word.
    """
    synsets = wordnet.synsets(word)
    print(f"Synsets for '{word}': {synsets}")
    return synsets

def definitions(word):
    """
    Print each synset and its corresponding definition for a given word.
    """
    synsets = Synsets(word)
    for syn in synsets:
        print(f"{syn.name()} - {syn.definition()}")

def path_based_similarity(word1, word2):
    """
    Calculate the highest path similarity score between any pair of synsets
    from the two given words and print it.
    """
    max_score = 0
    synsets1 = Synsets(word1)
    synsets2 = Synsets(word2)
    
    for syn1 in synsets1:
        for syn2 in synsets2:
            score = syn1.path_similarity(syn2)
            if score is not None and score > max_score:
                max_score = score

    print(f"Path-based similarity between '{word1}' and '{word2}': {max_score}")

# Example Usage
Synsets('progress')
Synsets('advance')
print("=" * 50)

definitions('progress')
definitions('advance')
print("=" * 50)

path_based_similarity('progress', 'advance')
