import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def Cross_validation(data, targets, tfidf, clf_cv, model_name):  # Performs cross-validation on SVC

    kf = KFold(n_splits=10, shuffle=True, random_state=9)  # 10-fold cross-validation
    scores = []
    data_train_list = []
    targets_train_list = []
    data_test_list = []
    targets_test_list = []
    iteration = 0
    print("Performing cross-validation for {}...".format(model_name))
    for train_index, test_index in kf.split(data):
        iteration += 1
        print("Iteration ", iteration)
        data_train_cv, targets_train_cv = data[train_index], targets[train_index]
        data_test_cv, targets_test_cv = data[test_index], targets[test_index]
        data_train_list.append(data_train_cv)  # appending training data for each iteration
        data_test_list.append(data_test_cv)  # appending test data for each iteration
        targets_train_list.append(targets_train_cv)  # appending training targets for each iteration
        targets_test_list.append(targets_test_cv)  # appending test targets for each iteration
        tfidf.fit(data_train_cv)  # learning vocabulary of training set
        data_train_tfidf_cv = tfidf.transform(data_train_cv)
        print("Shape of training data: ", data_train_tfidf_cv.shape)
        data_test_tfidf_cv = tfidf.transform(data_test_cv)
        print("Shape of test data: ", data_test_tfidf_cv.shape)
        clf_cv.fit(data_train_tfidf_cv, targets_train_cv)  # Fitting SVC
        score = clf_cv.score(data_test_tfidf_cv, targets_test_cv)  # Calculating accuracy
        scores.append(score)  # appending cross-validation accuracy for each iteration
    print("List of cross-validation accuracies for {}: ".format(model_name), scores)
    mean_accuracy = np.mean(scores)
    print("Mean cross-validation accuracy for {}: ".format(model_name), mean_accuracy)
    print("Best cross-validation accuracy for {}: ".format(model_name), max(scores))
    max_acc_index = scores.index(max(scores))  # best cross-validation accuracy
    max_acc_data_train = data_train_list[max_acc_index]  # training data corresponding to best cross-validation accuracy
    max_acc_data_test = data_test_list[max_acc_index]  # test data corresponding to best cross-validation accuracy
    max_acc_targets_train = targets_train_list[
        max_acc_index]  # training targets corresponding to best cross-validation accuracy
    max_acc_targets_test = targets_test_list[
        max_acc_index]  # test targets corresponding to best cross-validation accuracy

    return mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test


def normalizer(tweet):
    soup = BeautifulSoup(tweet, 'lxml')  # removing HTML encoding such as ‘&amp’,’&quot’
    souped = soup.get_text()
    only_words = re.sub("(@[A-Za-z0-9]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ",
                        souped)  # removing @mentions, hashtags, urls

    tokens = nltk.word_tokenize(only_words)
    removed_letters = [word for word in tokens if len(word) > 2]
    lower_case = [lwr.lower() for lwr in removed_letters]

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas


def c_matrix(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, tfidf, targets, clf,
             model_name):  # Creates Confusion matrix for SVC
    tfidf.fit(max_acc_data_train)
    max_acc_data_train_tfidf = tfidf.transform(max_acc_data_train)
    max_acc_data_test_tfidf = tfidf.transform(max_acc_data_test)
    clf.fit(max_acc_data_train_tfidf, max_acc_targets_train)  # Fitting SVC
    targets_pred = clf.predict(max_acc_data_test_tfidf)  # Prediction on test data
    conf_mat = confusion_matrix(max_acc_targets_test, targets_pred)
    print("Confusion matrix")
    print('TP: ', conf_mat[1, 1])
    print('TN: ', conf_mat[0, 0])
    print('FP: ', conf_mat[0, 1])
    print('FN: ', conf_mat[1, 0])
    d = {'High': 1, 'Low': 0}
    sentiment_df = targets.drop_duplicates().sort_values()
    sentiment_df = sentiment_df.apply(lambda x: d[x])
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=sentiment_df.values, yticklabels=sentiment_df.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix (Best Accuracy) - {}".format(model_name))
    plt.show()


def SVC_Save(data, targets, tfidf):
    tfidf.fit(data)  # learn vocabulary of entire data
    data_tfidf = tfidf.transform(data)
    pd.DataFrame.from_dict(data=dict([word, i] for i, word in enumerate(tfidf.get_feature_names())),
                           orient='index').to_csv('vocabulary_SVC.csv', header=False)
    print("Shape of tfidf matrix for saved SVC Model: ", data_tfidf.shape)
    clf = LinearSVC().fit(data_tfidf, targets)
    joblib.dump(clf, 'svc.sav')


def NBC_Save(data, targets, tfidf):
    tfidf.fit(data)  # learn vocabulary of entire data
    data_tfidf = tfidf.transform(data)
    pd.DataFrame.from_dict(data=dict([word, i] for i, word in enumerate(tfidf.get_feature_names())),
                           orient='index').to_csv('vocabulary_NBC.csv', header=False)
    print("Shape of tfidf matrix for saved NBC Model: ", data_tfidf.shape)
    clf = MultinomialNB(alpha=1.0).fit(data_tfidf, targets)
    joblib.dump(clf, 'nbc.sav')


def mainCreation():
    #  open up your csv file with the sentiment results
    with open('MedReviews.csv', 'r', encoding='ISO-8859-1') as csvfile:
        df = pd.read_csv(csvfile)
    df['normalized_review'] = df.Review.apply(normalizer)
    pd.set_option('display.max_colwidth', -1)
    df = df[df['normalized_review'].map(len) > 0]  # removing rows with normalized tweets of length 0
    print("Printing top 5 rows of dataframe showing original and cleaned tweets....")
    print(df[['Review', 'normalized_review']].head())
    df.drop(['Medicine', 'Condition', 'Review'], axis=1, inplace=True)
    # Saving cleaned tweets to csv
    df.to_csv('Cleaned_MedReview_Data.csv', encoding='utf-8', index=False)

    # Reading cleaned tweets as dataframe
    dataset = pd.read_csv("Cleaned_MedReview_Data.csv", encoding="ISO-8859-1")
    pd.set_option('display.max_colwidth', -1)
    data = dataset.normalized_review
    targets = dataset.Rating
    Y_SMOTE = targets

    pd.value_counts(pd.Series(Y_SMOTE)).plot.bar()
    plt.title('Rating class histogram')
    plt.xlabel('Rating Class')
    plt.ylabel('Frequency')
    plt.show()

    tv = TfidfVectorizer(stop_words=None, max_features=100000)
    testing_tfidf = tv.fit_transform(data)

    # SMOTE
    print("Number of observations in each class before oversampling (training data): \n",
          pd.Series(Y_SMOTE).value_counts())

    smote = SMOTE(random_state=100)
    X_SMOTE, Y_SMOTE = smote.fit_sample(testing_tfidf, targets)

    print("Number of observations in each class after oversampling (training data): \n",
          pd.Series(Y_SMOTE).value_counts())

    pd.value_counts(pd.Series(Y_SMOTE)).plot.bar()
    plt.title('Rating class histogram')
    plt.xlabel('Rating Class')
    plt.ylabel('Frequency')
    plt.show()

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2))
    SVC_clf = LinearSVC()  # SVC Model
    SVC_mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test = Cross_validation(
        data, Y_SMOTE, tfidf, SVC_clf, "SVC")  # SVC cross-validation
    c_matrix(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, tfidf, Y_SMOTE,
             SVC_clf, "SVC")  # SVC confusion matrix

    NBC_clf = MultinomialNB()  # NBC Model
    NBC_mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test = Cross_validation(
        data, Y_SMOTE, tfidf, NBC_clf, "NBC")  # NBC cross-validation
    c_matrix(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, tfidf, Y_SMOTE,
             NBC_clf, "NBC")  # NBC confusion matrix

    if SVC_mean_accuracy > NBC_mean_accuracy:
        SVC_Save(data, targets, tfidf)
        print("SVC is best Model based on mean accuracy score")
    else:
        NBC_Save(data, targets, tfidf)
        print("NBC is best model based on mean accuracy score")


def mainDeployment():
    # Loading the saved model
    model = joblib.load('svc.sav')
    vocabulary_model = pd.read_csv('vocabulary_SVC.csv', header=None)
    vocabulary_model_dict = {}
    for i, word in enumerate(vocabulary_model[0]):
        vocabulary_model_dict[word] = i
    tfidf = TfidfVectorizer(sublinear_tf=True, vocabulary=vocabulary_model_dict, min_df=5, norm='l2',
                            ngram_range=(1, 3))  # min_df=5 is clever way of feature engineering

    # Reading test dataset as dataframe
    with open('NoRatings.csv', 'r', encoding='ISO-8859-1') as csvfile:
        df = pd.read_csv(csvfile)
    # Normalizing retrieved tweets
    df['NoRating_normalized_review'] = df.Review.apply(normalizer)
    pd.set_option('display.max_colwidth', -1)  # Setting this so we can see the full content of cells
    df = df[df['NoRating_normalized_review'].map(len) > 0]  # removing rows with normalized tweets of length 0
    print("Number of tweets remaining after cleaning: ", df.NoRating_normalized_review.shape[0])
    print(df[['Review', 'NoRating_normalized_review']].head())
    # Saving cleaned tweets to csv file
    df.drop(['Medicine', 'Condition', 'Review'], axis=1, inplace=True)
    df.to_csv('NoRating_cleaned_reviews.csv', encoding='utf-8', index=False)

    cleaned_tweets = pd.read_csv("NoRating_cleaned_reviews.csv", encoding="ISO-8859-1")
    pd.set_option('display.max_colwidth', -1)
    cleaned_ratings_tfidf = tfidf.fit_transform(cleaned_tweets['NoRating_normalized_review'])
    targets_pred = model.predict(cleaned_ratings_tfidf)
    # Saving predicted Ratings of tweets to csv
    cleaned_tweets['predicted_ratings'] = targets_pred.reshape(-1, 1)
    cleaned_tweets.to_csv('NoRating_predicted_reviews.csv', encoding='utf-8', index=False)


mainCreation()
mainDeployment()
