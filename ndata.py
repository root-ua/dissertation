from numpy import zeros, append, save, column_stack, array
from scipy.io import savemat
from nltk import word_tokenize, pos_tag
import os



# Create arrays
adjectives = []
adverbs = []
adjectives_count = []
adverbs_count = []
adjectives_result = zeros((0, 0), dtype='u4')
adverbs_result = zeros((0, 0), dtype='u4')
negative_adjectives_result = zeros((0, 0), dtype='u4')

# emotions: 0 - negative, 1 - positive (+1 for MATLAB)
# fairness: 0 - deceptive, 1 - truthful (+1 for MATLAB)
# ['review text', positive/negative, truthful/deceptive]
reviews = zeros((0, 3))

# Dictionary with negating words for adjectives
reject = [
    # (?) JJ
    ["wasnt", "isnt", "arent", "dont", "not", "wont", "nt"],
    # (?) (X) JJ
    ["couldnt", "shouldnt", "doesnt", "didnt", "wouldnt"],
    # (?) (not) (X) JJ
    ["could not", "should not", "does not", "did not", "would not",
     "is not", "was not"]
]

# Read dataset directories recursively
def read_folders(path):
    emotions = {0: 'negative_polarity', 1: 'positive_polarity'}
    fairness = {0: 'deceptive_from_MTurk', 1: 'truthful_from_Web'}
    for e in range(0, 2, 1):
        for f in range(0, 2, 1):
            temp_path = path + emotions[e] + "\\" + fairness[f] + "\\"
            for p in range(1, 6):
                fin_path = temp_path + "fold" + p.__str__() + "\\"
                read_reviews(fin_path, e + 1, f + 1)


# Read all reviews from entire folder
def read_reviews(folder, em, fair):
    global reviews, adjectives_result, adverbs_result, \
        negative_adjectives_result
    for f in os.listdir(folder):
        full_file = os.path.join(folder, f)
        if os.path.isfile(full_file):
            file_pointer = open(full_file, 'r')
            text = file_pointer.read().rstrip()
            reviews = append(reviews, [[text, em, fair]], axis=0)

            review_index = len(reviews) - 1
            if review_index < 0:
                review_index = 0

            adjectives_result = append(adjectives_result,
                                       [zeros(adjectives_result.shape[1])],
                                       axis=0)
            negative_adjectives_result = append(negative_adjectives_result,
                                                [zeros(negative_adjectives_result.shape[1])],
                                                axis=0)
            adverbs_result = append(adverbs_result, [zeros(adverbs_result.shape[1])],
                                    axis=0)

            text_tokens = word_tokenize(text)
            text_pos = pos_tag(text_tokens)
            for (tx, tk) in text_pos:
                tx = str.lower(tx).strip("'")
                if tk[0:2] == 'RB':
                    add_pos(0, tx, review_index, text_tokens)
                elif tk[0:2] == 'JJ':
                    add_pos(1, tx, review_index, text_tokens)


# Add feature to the array
# 0 - adverb; 1 - adjective
def add_pos(pos, value, review_id, tokens):
    global adverbs, adjectives, adverbs_result, adjectives_result, \
        negative_adjectives_result
    if pos == 0:
        try:
            txi = adverbs.index(value)
        except ValueError:
            txi = -1

        if txi < 0:
            adverbs.append(value)
            adverbs_count.append(1)
            adverbs_result = append(adverbs_result,
                                    zeros([len(adverbs_result), 1]),
                                    axis=1)
            adverbs_result[review_id][adverbs.index(value)] = 1
        else:
            adverbs_count[adverbs.index(value)] += 1
            adverbs_result[review_id][adverbs.index(value)] += 1

        return adverbs.index(value)
    else:
        try:
            txi = adjectives.index(value)
        except ValueError:
            txi = -1

        if txi < 0:
            adjectives.append(value)
            adjectives_count.append(1)
            adjectives_result = append(adjectives_result,
                                       zeros([len(adjectives_result), 1]),
                                       axis=1)
            adjectives_result[review_id][adjectives.index(value)] = 1
            negative_adjectives_result = append(negative_adjectives_result,
                                                zeros([len(negative_adjectives_result),
                                                       1]), axis=1)
            negative_adjectives_result[review_id][adjectives.index(value)] = \
                find_negative(tokens, value)
        else:
            adjectives_count[adjectives.index(value)] += 1
            adjectives_result[review_id][adjectives.index(value)] += 1
            negative_adjectives_result[review_id][adjectives.index(value)] = \
                find_negative(tokens, value)


# Find negated adjectives in text
def find_negative(tokens, adj):
    negatives = 0
    for (i, v) in enumerate(tokens):
        if str.lower(v) == str.lower(adj):
            if i >= 1:
                word = str.lower(tokens[i - 1]).strip("'").strip('"').\
                    replace("'", "").replace('"', '')
                if word in reject[0]:
                    negatives += 1
                    continue
            if i >= 2:
                word = str.lower(tokens[i - 2]).strip("'").strip('"').\
                    replace("'", "").replace('"', '')
                if word in reject[1]:
                    negatives += 1
                    continue
            if i >= 3:
                word = str.lower(tokens[i - 3]).strip("'").strip('"').\
                    replace("'", "").replace('"', '')
                word += ' ' + str.lower(tokens[i - 2]).strip("'").\
                    strip('"').replace("'", "").replace('"', '')
                if word in reject[2]:
                    negatives += 1
                    continue
    return negatives


# Save obtained results in different formats
def save_data(temp, arrays_to_file, arrays_to_separate_matfiles):
    if arrays_to_file == 1:
        d = 'npy/' + temp
        save(d + 'reviews.npy', reviews)
        save(d + 'adverbs.npy', adverbs)
        save(d + 'adjectives.npy', adjectives)
        save(d + 'adverbs_result.npy', adverbs_result)
        save(d + 'adjectives_result.npy', adjectives_result)
        save(d + 'negative_adjectives_result.npy',
             negative_adjectives_result)

    if arrays_to_separate_matfiles == 1:
        d = 'mats/' + temp
        savemat(d + 'reviews.mat',
                mdict={'reviews': reviews_modified})
        savemat(d + 'adverbs_result.mat',
                mdict={'adverbs_result': adverbs_result})
        savemat(d + 'adjectives_result.mat',
                mdict={'adjectives_result': adjectives_result})
        savemat(d + 'negative_adjectives_result.mat',
                mdict={'negative_adjectives_result': negative_adjectives_result})

# ## MAIN RUN SECTION

read_folders("E:\\op_spam_v1.4\\")

reviews_modified = zeros((0, 1), dtype='u4')
reviews_modified = reviews[0:, 1:2].astype('u4')

### SAVE RESULTS
save_data('', 1, 1)
