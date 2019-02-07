import collections
# importing all the resources from nltk package
from nltk import *
from itertools import chain

# reading the data from 'input.txt' file
input = open("input.txt", encoding='utf-8').read()
# tokenizing the text into words list
words = word_tokenize(input)
# tokenizing the text into sentences list
sentences = sent_tokenize(input)

print("-----------Lemmatization------")
# WordNet Lemmatizer- Lemmatize using WordNet’s built-in morphy function.
# Returns the input word unchanged if it cannot be found in WordNet.
lemmatizer = WordNetLemmatizer()
for word in words:
    # ignoring the special characters as word tokenizing considers special characters as words
    if not re.search("[($@!:.,')]", word):
        print(word + ":" + lemmatizer.lemmatize(word))

# applying the bigram on the text
# printing the list of bigrams
bi_grams = list(ngrams(words,2))
print("Bigrams: ", bi_grams)

# calculating the bigram frequency
bigram_COUNTS = collections.Counter(bi_grams)
print("Bigrams Frequency: ", bigram_COUNTS)
# getting the top five bigrams repeated most
top_five = bigram_COUNTS.most_common(5)
print("Top Five Bigrams: ",top_five)
#sentences with the most repeated bi-grams
# calculating the most repeated bigrams
most_repeated_bigrams = FreqDist(list(ngrams(words,2))).most_common()
most = [i[0] for i in most_repeated_bigrams]
# getting all the sentences with the most repeated bi-grams
conc_sentence = ""
for sentence in sentences:
    for word in chain(*most):
        if word in sentence and not re.search("[($@!:.,'^-_)]", word):
            conc_sentence =  conc_sentence+sentence
print("Concatenated Sentences: ", conc_sentence)







