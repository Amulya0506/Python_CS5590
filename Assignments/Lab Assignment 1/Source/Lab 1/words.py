def middle_word(words_1):
    length = len(words_1)  # gives the number of words in a sentence
    if length % 2 == 0:  # if the sentence has even number of words
        word1 = words[length // 2 - 1]
        word2 = words[(length // 2)]
        print("Middle words are: [%s, %s]" % (word1, word2))
    else:  # if the sentence has odd number of words
        print("Middle word is: ", words[(length - 1) // 2])


def longest_word(words_2):
    word = max(words_2, key=len)  # based on length it gives longest word in list of words
    print("Longest word is :", word)


def rev_words(words_3):
    # reverses each word and join words with space & converted to lower case
    print("Sentence with reverse words is:",
          ' '.join(word[::-1].lower() for word in words_3))


if __name__ == '__main__':
    sentence = input("Enter the sentence: ")
    words = sentence.split()
    middle_word(words)  # first comes here and calls 'middle_word(x)' function
    longest_word(words)
    rev_words(words)
