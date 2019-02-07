# Get the sentence from user
sentence = input("Enter the sentence: ")
def word_count(sentence):

    counts = dict()
    words = sentence.split()
    for word in words:
        if word in counts:
            counts[word] = counts[word] + 1
        else:
            counts[word] = 1
    sortedResult = sorted(counts.items())
    result = dict(sortedResult)
    return result

print(word_count(sentence))