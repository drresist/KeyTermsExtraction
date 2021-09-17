from lxml import etree
import nltk

# nltk.download()

"""
Read an XML file containing news stories and headlines.
Extract the headers and the text.
Tokenize the texts.
For each story, find the most frequent tokens that appear in it.

Print the headline of each news story the five most frequent tokens in descending order. 
Take a look at a sample below. 
Also, display the titles and keywords in the same order they are presented in the file.
"""


# Read an XML file containing news stories and headlines.
# Extract the headers and the text.
def read_file(filename) -> dict:
    xml_tree = etree.parse(filename).getroot()
    headers = [news[0].text for news in xml_tree[0]]
    text = [news[1].text for news in xml_tree[0]]
    
    return dict(zip(headers, text))


# Tokenize the texts.
def tokenize_text(dictionary: dict) -> dict:
    tokens = []
    for _, text in dictionary.items():
        tokens.append(nltk.tokenize.word_tokenize(str(text).lower()))
    return dict(zip(dictionary.keys(), tokens))


# For each story, find the most frequent tokens that appear in it.
def count_frequency(dictionary: dict) -> dict:
    freq_dict = []
    for _, tokens in dictionary.items():
        tokens = sorted(tokens, reverse=True)
        fdist = nltk.FreqDist(tokens)
        freq_dict.append([i[0] for i in fdist.most_common(5)])
    return dict(zip(dictionary.keys(), freq_dict))


# Print the headline of each news story the five most frequent tokens in descending order.
def print_output(dictionary: dict):
    for header, most_frequent_word in dictionary.items():
        print(header + ":")
        print(" ".join(most_frequent_word))



def main():
    header_text = read_file(filename="news.xml")  # {[headers] : [text]}
    header_token = tokenize_text(header_text)  # {[headers]: [tokens]}
    header_most_common = count_frequency(header_token)  # {[headers]: [most common]}
    print_output(header_most_common)


if __name__ == "__main__":
    main()
