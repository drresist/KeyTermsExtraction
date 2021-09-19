import string
from lxml import etree
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd
# nltk.download('averaged_perceptron_tagger')
# nltk.download()

"""
In this stage, your program should:
Read an XML-file containing stories and headlines.
Extract the headers and the text.
Tokenize each text.
Lemmatize each word in the story.
Get rid of punctuation, stopwords, and non-nouns with the help of NLTK.
Count the TF-IDF metric for each word in all stories.
Pick the five best scoring words.
"""

TEXT_LIST = []

# Read an XML file containing news stories and headlines.
# Extract the headers and the text.
def read_file(filename) -> dict:
    xml_tree = etree.parse(filename).getroot()
    headers = [news[0].text for news in xml_tree[0]]
    text = [news[1].text for news in xml_tree[0]]

    return dict(zip(headers, text))


# Tokenize the texts.
def tokenize_text(dictionary: dict) -> dict:
    tokens_list = []
    lemmatizer = nltk.WordNetLemmatizer()  # lemmatizer
    stop_words = stopwords.words('english') + list(string.punctuation)
    for header, text in dictionary.items():
        tokens = nltk.tokenize.word_tokenize(text.lower())
        lemma = [lemmatizer.lemmatize(x) for x in tokens]
        lemma = [x for x in lemma if x not in stop_words]
        pos_tags = [
            nltk.pos_tag([w])[0][0] for w in lemma if nltk.pos_tag([w])[0][1] == 'NN'
        ]
        tokens_list.append(pos_tags)
        TEXT_LIST.append(' '.join(pos_tags))
    return dict(zip(dictionary.keys(), tokens_list))


def tfidf_counter(dictionary:dict) -> dict:
    """
    Count the TF-IDF metric for each word in all stories.
    :param dictionary:
    :return:
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(TEXT_LIST)
    for title, row in zip(dictionary.keys(), tfidf_matrix):
        print(f"{title}:")
        tfidf_decoded = [
            (vectorizer.get_feature_names()[k[1]],v) for k,v in row.todok().items()
        ]
        alpha_sort = sorted(tfidf_decoded, key=lambda x:x[0], reverse=True) # by nouns alphabetically
        freq_sort = sorted(alpha_sort, key=lambda x:x[1], reverse=True) # by tf_idf score
        common_five = 0
        for token, freq in freq_sort:
            common_five += 1
            if common_five <= 5:
                print(token, end=' ')
            else:
                print('\n')
                break

# For each story, find the most frequent tokens that appear in it.
def count_frequency(dictionary: dict) -> dict:
    freq_dict = []
    for header, tokens in dictionary.items():
        tokens = sorted(tokens, reverse=True)
        fdist = nltk.FreqDist(tokens)
        freq_dict.append(
            [i[0] for i in fdist.most_common(5)]
        )
    return dict(zip(dictionary.keys(), freq_dict))


# # Print the headline of each news story the five most frequent tokens in descending order.
# def print_output(dictionary: dict):
#     for header, most_frequent_word in dictionary.items():
#         print(header + ':')
#         print(' '.join(most_frequent_word))
#     return 0


def main():
    header_text = read_file(filename='news.xml')  # {[headers] : [text]}
    header_token = tokenize_text(header_text)  # {[headers]: [tokens]}
    header_most_common = count_frequency(header_token)  # {[headers]: [most common]}
    # print_output(header_most_common)
    tfidf_counter(header_most_common)

if __name__ == '__main__':
    main()

