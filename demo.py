import random
from data_preparation import clean_string


PREFERENCES = ["business", "entertainment", "politics", "sport", "tech"]

class bcolors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def assign_preferences(users):
    users_with_preferences = {}
    for user in users:
        preferences = []
        possibilities = list(PREFERENCES)
        for pref in range(random.randint(1, 3)):
            category = random.choice(possibilities)
            preferences.append(category)
            possibilities.remove(category)
        users_with_preferences[user] = preferences
    return users_with_preferences


def predict_doc_type(doc, vect, model):
    doc_cleaned = clean_string(doc)
    corpus = []
    corpus.append(doc_cleaned)
    test_vect = vect.transform(corpus)
    return model.predict(test_vect)[0]


# Testing results
def run_demo(vect, model):
    users = ["Maria   ", "Nacho   ", "Luca    ", "Adam    ", "Tom     ", "Mike    "]
    users_with_preferences = assign_preferences(users)
    print("\n--------------------------------------------------------")
    print("------------------------- DEMO -------------------------")
    print("--------------------------------------------------------\n")

    print("--------------------------------------------------------\n")
    print("USER PREFERENCES:")
    for user in users_with_preferences:
        print(user, " -> ", users_with_preferences[user])
    print("\n--------------------------------------------------------")

    while True:
        try:
            print("---------------------------------------------------------------------------------------------------------------")
            test_corpus = input(bcolors.BOLD + "Paste here an article without newline characters (\\n) (Press 'q' to quit or 'r' to reassign the preferences):\n" + bcolors.ENDC)
            print("---------------------------------------------------------------------------------------------------------------")
        except ValueError:
            continue
        if test_corpus == "q":
            break
        if test_corpus == "r":
            users_with_preferences = assign_preferences(users)
            print("--------------------------------------------------------\n")
            print("USER PREFERENCES:")
            for user in users_with_preferences:
                print(user, "->  ", users_with_preferences[user])
            print("\n--------------------------------------------------------")
            continue
        else:

            result = predict_doc_type(test_corpus, vect, model)
            if(result == "business"):
                topic = bcolors.BOLD + bcolors.YELLOW + result.upper() + bcolors.ENDC
            if (result == "entertainment"):
                topic = bcolors.BOLD + bcolors.RED + result.upper() + bcolors.ENDC
            if (result == "politics"):
                topic = bcolors.BOLD + bcolors.PURPLE + result.upper() + bcolors.ENDC
            if (result == "sport"):
                topic = bcolors.BOLD + bcolors.BLUE + result.upper() + bcolors.ENDC
            if (result == "tech"):
                topic = bcolors.BOLD + bcolors.GREEN + result.upper() + bcolors.ENDC
            print("--------------------------------------------------------\n")
            print(bcolors.BOLD + "This article talks about "+ topic + bcolors.BOLD + " and it's addressed to:"+bcolors.ENDC)
            for user in users_with_preferences:
                if result in users_with_preferences[user]:
                    print(user)
            print("\n--------------------------------------------------------")
            continue