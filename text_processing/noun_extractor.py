'''
Modified version
'''

from __future__ import division  # To force result of two integers divided to be a floating point number

from collections import Counter
import math
from nltk import pos_tag_sents, word_tokenize, Text, BigramCollocationFinder, BigramAssocMeasures


def extract_nouns(text, language="english", proper_nouns_allowed=True):
    """
    Extracts all the words identified as nouns (singular and plural) in one continuous string of text.

    :param text: The text as one long string.
    :param language: A language supported by the NLTK tagger.
    :param proper_nouns_allowed: Whether or not to include proper nouns.
    :return: All words identified as nouns in a list.
    """
    tagged_sentences = pos_tag_sents([word_tokenize(text, language)])

    nouns = ["NN", "NNS"]
    proper_nouns = ["NNP", "NP"]
    if proper_nouns_allowed:
        nouns += proper_nouns

    results = []
    for tagged_tokens in tagged_sentences:
        for token, tag in tagged_tokens:
            if tag in nouns:
                results.append(token.lower())

    from tf_idf import stem_tokens
    #from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    #stemmer = SnowballStemmer("english")
    results = stem_tokens(results)

    return results


def frequency_count(word_list):
    """
    Counts the frequency of each word in a long list of strings.

    :param word_list: List of words as strings.
    :return: A counter object containing the count of each word.
    """
    frequencies = Counter(word_list)

    return frequencies


def collocations(text, language='english', num=0, window_size=2): #num=20
    """
    A reimplementation of the basic workings of the collocations method of the ``Text`` class of NLTK.

    :param text: raw text
    :param language: language to use to eliminate stopwords
    :param num: number of collocations
    :param window_size: window for collocations
    :return: a list of collocations for the text
    """

    from nltk.corpus import stopwords

    ignored_words = stopwords.words(language)
    finder = BigramCollocationFinder.from_words(word_tokenize(text), window_size)
    finder.apply_freq_filter(2)
    finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
    bigram_measures = BigramAssocMeasures()
    c = finder.nbest(bigram_measures.likelihood_ratio, num)

    collocation_strings = [w1 + ' ' + w2 for w1, w2 in c]
    return collocation_strings


def remove_stopwords(words, language="english"):
    """
    Removes all stopwords from a list of words.

    :param words: The words to be filtered
    :param language: The language of the words
    :return: A list of the remaining words (non-stopwords)
    """
    from nltk.corpus import stopwords

    ignored_words = stopwords.words(language)

    return [word for word in words if word.lower() not in ignored_words]


def combine_collocations(count_up_to_cutoff, words_to_be_retrieved, text, language="english", collocations_number=0, collocations_window_size=2):
    """
    Finds the collocations in a text and combines the entries in a supplied list to reflect the collocations. Also reduces the total count in order to reflect the combined collocations.

    :param count_up_to_cutoff: The sum of the counts in the list
    :param words_to_be_retrieved: A list of tuples of form [(count, word), ...]
    :param text: The text that the words have been retrieved from.
    :param language: The language the text is written in
    :return: A list of tuples of form [(count, word), ...] with collocations combined
    """
    words_that_occur_together = collocations(text, language=language, num=collocations_number, window_size=collocations_window_size) #, num=count_up_to_cutoff)
    total_count = count_up_to_cutoff

    for c in words_that_occur_together:
        w1, w2 = c.lower().split()

        index_of_w1, index_of_w2 = -1, -1

        for i in range(len(words_to_be_retrieved)):
            _, word = words_to_be_retrieved[i]
            if w1 == word:
                index_of_w1 = i
                break

        for j in range(len(words_to_be_retrieved)):
            _, word = words_to_be_retrieved[j]
            if w2 == word:
                index_of_w2 = j
                break

        if index_of_w1 > -1 and index_of_w2 > -1:
            count_1, _ = words_to_be_retrieved[index_of_w1]
            count_2, _ = words_to_be_retrieved[index_of_w2]

            most_used_count = max(count_1, count_2)
            least_used_count = min(count_1, count_2)

            # Make sure to delete the last index first (or else the list would be skewed before deletion)
            if index_of_w1 < index_of_w2:
                words_to_be_retrieved.pop(index_of_w2)
                words_to_be_retrieved.pop(index_of_w1)
            else:
                words_to_be_retrieved.pop(index_of_w1)
                words_to_be_retrieved.pop(index_of_w2)

            # Add the collocation back into the list with the least of the two counts.
            words_to_be_retrieved.append((least_used_count, c))
            # Remove the maximum count in order to make total count correct.
            total_count -= most_used_count

    words_to_be_retrieved.sort(reverse=True)

    return total_count, words_to_be_retrieved


def list_queries_from_text(text, cutoff_percentage=0.25, number_of_images_to_be_retrieved=500, language='english',
                           proper_nouns_allowed=True):  # pragma: no cover
    """
    Takes a text and performs a frequency analysis of the nouns of the text.

    The words representing the a certain share of all noun occurrences are returned in a list of tuples.

    The tuples returned are of the form (word, images_to_be_retrieved_for_the_word), where the number of images to be
    retrieved is calculated based on the word's share of occurrences in the top percentile and the total number of
    images that will be retrieved.

    :param text: The text to be analysed
    :param cutoff_percentage: The share of words at which the nouns will be cut off.
    :param number_of_images_to_be_retrieved: The number of images that will be retrieved from some engine.
    :param language: The language of the text. Used for stopword elimination.
    :param proper_nouns_allowed: Whether or not to extract proper nouns.
    :return: A list of tuples of the form (word, images_to_be_retrieved_for_the_word)
    """

    nouns = extract_nouns(text, language=language, proper_nouns_allowed=proper_nouns_allowed)

    non_stop_nouns = remove_stopwords(nouns, language)

    frequencies = frequency_count(non_stop_nouns)

    sum_of_frequencies, words_sorted_by_frequency = sort_words_by_frequencies(frequencies)

    count_up_to_cutoff, words_to_be_retrieved = _get_top_percentage_of_words(cutoff_percentage, sum_of_frequencies,
                                                                             words_sorted_by_frequency)

    new_count, words_with_collocations_combined = combine_collocations(count_up_to_cutoff, words_to_be_retrieved, text, language=language)

    query_list = get_queries(new_count, number_of_images_to_be_retrieved, words_with_collocations_combined)

    return query_list


def get_queries(sum_of_word_counts, number_of_images_to_be_retrieved, words_to_be_retrieved):
    """
    Get a list that can be used to query some image search engine for the given words. The total number of images to
    retrieve should be less than or equal to the number_of_images_to_be_retrieved.

    The queries are distributed proportionally to the words' counts.

    :param sum_of_word_counts: The sum of word counts
    :param number_of_images_to_be_retrieved: The number of images to be retrieved by the queries.
    :param words_to_be_retrieved: A list of tuples of form [(count, word)]
    :return: A list of tuples of form [(query, number_of_results)]
    """
    images_per_occurrence = number_of_images_to_be_retrieved / sum_of_word_counts
    query_list = []
    for count, word in words_to_be_retrieved:
        number_of_results_in_query = math.floor(count * images_per_occurrence)

        query_list.append((word, number_of_results_in_query))
    return query_list


def _get_top_percentage_of_words(cutoff_percentage, sum_of_frequencies, words_sorted_by_frequency):
    """
    Gets the top cutoff_percentage of occurrences of words.

    :param cutoff_percentage: The percentage to stop fetching at.
    :param sum_of_frequencies: The total sum of frequencies
    :param words_sorted_by_frequency: A list of tuples, [(count, word), ...], sorted by descending count.
    :return:
    """
    words_to_be_retrieved = []
    cutoff = math.floor(sum_of_frequencies * cutoff_percentage)
    count_up_to_cutoff = 0
    for pair in words_sorted_by_frequency:
        words_to_be_retrieved.append(pair)
        count_up_to_cutoff += pair[0]
        if count_up_to_cutoff >= cutoff:
            break  # Stop the loop when we have enough words
    return count_up_to_cutoff, words_to_be_retrieved


def sort_words_by_frequencies(frequencies, reverse=True):
    """
    Sort words by their frequencies. Return the sum of the frequencies and a list of tuples (freq, word).

    :param frequencies: A dictionary containing words and the number of occurrences
    :param reverse: Whether or not to sort descending.
    :return: sum of frequencies, words sorted by descending frequencies
    """

    sum_of_frequencies = 0
    words_sorted_by_frequency = []
    for key, value in frequencies.iteritems():
        sum_of_frequencies += value
        words_sorted_by_frequency.append((value, key))
    words_sorted_by_frequency.sort(reverse=reverse)
    return sum_of_frequencies, words_sorted_by_frequency
