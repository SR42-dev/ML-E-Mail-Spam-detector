'''

READ_ME :

Applicant details :

	Sriram R
	PES1UG20CS435
	sriram.radhakrishna42@gmail.com

Notes :

- Downloading email data from the internet as I have no spam in my inbox and the few that I do have contain a single image with no text

- Dataset link : https://github.com/OmkarPathak/Playing-with-datasets/blob/master/Email%20Spam%20Filtering/emails.csv

- predict function will not work without access to the lineSpace & getUsableText functions

- First execution will take more time than subsequent ones

References :

	https://www.udemy.com/course/machine-learning-course/learn/lecture/20822824#questions/12088156
	https://github.com/OmkarPathak/Playing-with-datasets/blob/master/Email%20Spam%20Filtering/emails.csv
	https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
	https://scikit-learn.org/stable/tutorial/basic/tutorial.html
	https://stackoverflow.com/questions/35360081/naive-bayes-vs-svm-for-classifying-text-data
	https://scikit-learn-general.narkive.com/fg8tBUr7/multinomialnb-vs-svm-svc-kernel-linear
	https://stackoverflow.com/questions/26693736/nltk-and-stopwords-fail-lookuperror
	https://www.programiz.com/python-programming/writing-csv-files
	https://medium.com/nerd-for-tech/probability-and-machine-learning-570815bad29d
	https://randerson112358.medium.com/email-spam-detection-using-python-machine-learning-abe38c889855
	https://towardsdatascience.com/how-i-used-machine-learning-to-classify-emails-and-turn-them-into-insights-efed37c1e66

'''

import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

path1 = 'emailsV1.1.csv' # sample dataset downloaded from the internet

def lineSpace() :
    print()
    print('_______________________________________________________________________________________')
    print()

def getUsableText(text) : # function to extract workable list of words

    nopunc = [char for char in text if char not in string.punctuation] # removing punctuation like !, ", #, $, %, &, ', (, ), *, +, -, ., /, :, ;, <, =, >, ?, @, [, \, ], ^, _, `, {, |, }, ~, etc.
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')] # removing useless words like “a”, “the”, “is”, “are”, etc (stopwords)
    return clean_words

def predict(path) : # not to be used as standalone without lineSpace & getUsableText in the same file

    ds = pd.read_csv(path)
    ds.head(5)

    lineSpace()

    print('Raw data dimensions : ',ds.shape)
    print('Column names : ',str(ds.columns)[7:21])
    ds.drop_duplicates(inplace = True) # removing duplicates
    print('Dimensions after duplicate drop : ',ds.shape)

    lineSpace()

    nltk.download('stopwords') # downloading stopwords into a variable 'stopwords'

    ds['text'].head().apply(getUsableText) # getting useful words from text column
    messages_bow = CountVectorizer(analyzer=getUsableText).fit_transform(ds['text']) # frequency vector of all unique words in all emails

    X_train, X_test, y_train, y_test = train_test_split(messages_bow, ds['spam'], test_size = 0.20, random_state = 0) # 80-20 split between training & testing data

    classifier = MultinomialNB() # baye's classifier (probabilistic)
    classifier.fit(X_train, y_train)

    lineSpace()

    pred = classifier.predict(X_train) # training the model
    print('Classification report (training) : \n\n', classification_report(y_train ,pred ))
    print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
    print('Accuracy: ', accuracy_score(y_train,pred))

    lineSpace()

    pred = classifier.predict(X_test) # testing the model
    print('Classification report (testing) : \n\n', classification_report(y_test ,pred))
    print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
    print('Accuracy: ', accuracy_score(y_test,pred))

    lineSpace()

predict(path1) # main execution
