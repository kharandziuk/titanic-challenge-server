from fuzzywuzzy import fuzz
from model import clf, lookup_list


# I use simple fuzzy match for string. Also I sort tokens instead of simple fuzzy match and I removed all the titles from the names
def lookup_func(name):
    match = max(lookup_list, key=lambda d: fuzz.token_sort_ratio(d[1], name))
    match_name = match[1]
    match_data = [match[0]] + match[2:]
    return match_name, match_data

def predict(vector):
    probability, _ = clf.predict_proba([vector])[0]
    return probability
