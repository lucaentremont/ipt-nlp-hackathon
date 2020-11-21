##
# NLP Hackathon Task 1 by:
# Isha Gupta
# Moritz Reihs
# Luca Entremont
##

## SETTINGS
clean_data = False
model_train = False
model_train2 = False
model_evaluate = False
model_train3 = True
model_evaluate3 = True


import pandas as pd
import os 
import spacy
#from nltk.corpus import stopwords
#german_stop_words = stopwords.words('german')
dir_path = os.path.dirname(os.path.realpath(__file__))
# hallo moritz
# from nltk.corpus import stopwords 
# from textblob import TextBlob, Word
from langdetect import detect

if clean_data:
    ## 1 -- Data Import
    uglyDataPath = "../data/test_reduced.csv"

    df = pd.read_csv(uglyDataPath, sep=";", encoding="utf8")
    df = df.dropna(subset=['MailTextBody'])
    df = df.reset_index(drop=True)
    df['Combined_Content'] = df['MailSubject'] +" "+ df['MailTextBody']
    df = df.sort_values(by='Id', ascending=False)
    #df = df.to_string()
    #print(df.head(5))

    # Import Stopwords

    german_stop_words = [x.strip() for x in open('/Users/luca/Offline/ipt-hackathon/german_stopwords.txt', 'r').readlines()]
    first_names = [x.strip() for x in open('/Users/luca/Offline/ipt-hackathon/first_names.all.txt', 'r').readlines()]
    last_names = [x.strip() for x in open('/Users/luca/Offline/ipt-hackathon/last_names.all.txt', 'r').readlines()]
    #print(german_stop_words)

    ## 2 -- Data Cleaning

    # Lower casing and removing punctuations 
    df['Combined_Content'] = df['Combined_Content'].apply(lambda x: " ".join(x.lower() for x in str(x).split())) 
    df['Combined_Content'] = df['Combined_Content'].str.replace('[^\w\s]','') 

    #print("After lowercase:\n ", df['Combined_Content'].head(5))
    N = 0
    def excluded(x):
        global N
        N= N+1
        if N %100 == 0:
            print(f"EXCLUDE STEP: {N}\n")
        fr = False
        perc_num =  len([1 for a in x if a.isdigit()])/len(x)
        bools = [x in german_stop_words,
                x in first_names,
                x in last_names,
                "md5" in x, 
                "http" in x,
                "cidimage" in x,
                "cid" in x,
                "inc" in x,
                "inc" in x,
                "mxm" in x,
                "_" == x,
                "gif" in x,
                x.isnumeric(),
                len(x) > 24,
                len(x) < 3,
                fr,
                perc_num >= .4]
        return bool(sum(bools))
    nlp = spacy.load('de_core_news_sm')
    #from spellchecker import SpellChecker
    #from autocorrect import Speller
    #spell = Speller(lang='de')
    #spell = SpellChecker(language='de')
    M = 0
    def lemm_spell(x):
        global M
        x = ' '.join([y for y in x.split(' ') if not(y in german_stop_words)])
        M= M+1
        if M %10 == 0:
            print(f"LEMM SPELL STEP: {M}\n")
        x.replace('_', ' ')
        #x = " ".join([spell.correction(a).lower() for a in x.split(' ')])
        doc = nlp(x)
        result = ' '.join(x.lemma_ for x in doc) 
        return result
        '''
        return ' '.join([spell.correction(w) for w in x.split(' ')])
        '''
    df['Combined_Content'] = df['Combined_Content'].apply(lambda x: " ".join(x for x in str(x).split() if not(excluded(x))))
    df['Combined_Content'] = df['Combined_Content'].apply(lambda x: lemm_spell(x))
    from langdetect import detect
    def lang_it(x):
        bl = False
        try:
            bl = (detect(x) == "de")
        except:
            bl = False
        return bl
    df[df['Combined_Content'].map(lambda x: lang_it(x))]
    print("After stopwords:\n ", df['Combined_Content'])


    df.to_csv("/Users/luca/Offline/data/test_reduced_clean.csv")
    #Lemmatisation

if model_train:
    import matplotlib.pyplot as plt
    import numpy as np 
    import pandas as pd 
    import matplotlib.pyplot as plt 
    import string 
    from nltk.stem import SnowballStemmer 
    from nltk.corpus import stopwords 
    from sklearn.feature_extraction.text import TfidfVectorizer 
    from sklearn.model_selection import train_test_split 
    import os 
    from textblob import TextBlob 
    from sklearn.metrics import confusion_matrix
    from nltk.stem import PorterStemmer 
    from textblob import Word 
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
    import sklearn.feature_extraction.text as text 
    from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm 
    from sklearn.naive_bayes import MultinomialNB 
    from sklearn.linear_model import LogisticRegression 
    from sklearn.ensemble import RandomForestClassifier 
    from sklearn.svm import LinearSVC 
    from sklearn.model_selection import cross_val_score 
    from io import StringIO 
    import seaborn as sns


    cleanedDataPath = "/Users/luca/Offline/data/clean.csv"
    df = pd.read_csv(cleanedDataPath, sep=",", encoding="utf8")
    df = df.reset_index(drop=True)
    df = df.dropna(subset=['ServiceProcessed'])
    #df = df.sort_values(by='Id', ascending=False)
    df = df.applymap(str)

    #fig = plt.figure(figsize=(8,6)) 
    #df.groupby('ServiceProcessed').Combined_Content.count().plot.bar(ylim=0) 
    #plt.show()


    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['Combined_Content'], df['ServiceProcessed'])

    encoder = preprocessing.LabelEncoder() 
    train_y = encoder.fit_transform(train_y) 
    valid_y = encoder.fit_transform(valid_y)
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=20000) 
    tfidf_vect.fit(df['Combined_Content']) 
    xtrain_tfidf = tfidf_vect.transform(train_x) 
    xvalid_tfidf = tfidf_vect.transform(valid_x)

    model = linear_model.LogisticRegression().fit(xtrain_tfidf, train_y)
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=2, penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)

    # Checking accuracy 
    accuracy = metrics.accuracy_score(model.predict(xvalid_tfidf), valid_y) 
    print("Accuracy: ", accuracy)

#conf_mat = confusion_matrix(valid_y, model.predict(xvalid_tfidf))

#category_id_df = df[['ServiceProcessed', 'Id']].drop_duplicates().sort_values('Id') 
# category_to_id = dict(category_id_df.values) 
# id_to_category = dict(category_id_df[['Id', 'ServiceProcessed']].values)
# fig, ax = plt.subplots(figsize=(8,6)) 
# sns.heatmap(conf_mat, annot=True, fmt='d', cmap="BuPu", xticklabels=category_id_df[['ServiceProcessed']].values, yticklabels=category_id_df[['ServiceProcessed']].values) 
# plt.ylabel('Actual') 
# plt.xlabel('Predicted') 
# plt.show()


## NN Solution
## Model Definition
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# Import Libraries 
if model_train2:
    import sys, os, re, csv, codecs
    import numpy as np
    import pandas as pd
    import keras
    from keras.preprocessing.text import Tokenizer 
    from keras.preprocessing.sequence import pad_sequences 
    from keras.utils import to_categorical 
    from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation 
    from keras.layers import Bidirectional, GlobalMaxPool1D, Conv1D, SimpleRNN 
    from keras.models import Model 
    from keras.models import Sequential 
    from keras import initializers, regularizers, constraints, optimizers, layers 
    from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization 
    from keras.layers import Conv1D, MaxPooling1D, Embedding
    from keras.models import Sequential
    from sklearn.metrics import confusion_matrix
    import itertools

    import matplotlib.pyplot as plt 
    from sklearn.model_selection import train_test_split
    labels = ["EDA_ANW_Intranet/Collaboration EDA", 
    "EDA_ANW_SAP Services",
    "EDA_ANW_SysPeDoc",
    "EDA_S_APS_Monitor",
    "EDA_S_APS_OS_BasisSW",
    "EDA_S_APS_PC",
    "EDA_S_APS_Peripherie",
    "EDA_S_BA_2FA",
    "EDA_S_BA_Account",
    "EDA_S_BA_Mailbox",
    "EDA_S_BA_UCC_Benutzertelefonie",
    "EDA_S_BA_UCC_IVR",
    "EDA_S_Benutzerunterstützung",
    "EDA_S_Betrieb Übermittlungssysteme",
    "EDA_S_Mobile Kommunikation",
    "EDA_S_Netzdrucker",
    "EDA_S_Order Management",
    "EDA_S_Peripheriegeräte",
    "EDA_S_Zusätzliche Software"]
    #combined_into_others = ["EDA_S_BA_Datenablage", "EDA_S_BA_Internetzugriff", "EDA_S_BA_RemoteAccess", "EDA_S_IT Sicherheit", "EDA_S_Netzwerk Ausland", "EDA_S_Raumbewirtschaftung"]
    others_name = "EDA_others"
    def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    NBs_list =[14000] #[int(x) for x in np.linspace(10000,20000,10)]
    DIMs_list = [200]#[int(x) for x in np.linspace(800,1200,5)]
    NBs_acc = []
    DIMs_acc = []
    for iter_DIM in DIMs_list:

        cleanedDataPath = "/Users/luca/Offline/data/clean.csv"
        df = pd.read_csv(cleanedDataPath, sep=",", encoding="utf8")
        df = df.reset_index(drop=True)
        df = df.dropna(subset=['ServiceProcessed'])
        df = df.sort_values(by='Id', ascending=False)
        df = df.applymap(str)
        #print(df['ServiceProcessed'].value_counts())
        #input()
        #infrequent = []
        infrequent = ["EDA_S_Raumbewirtschaftung", 
        "EDA_ANW_ZACWEB", 
        "EDA_S_BA_Datenablage", 
        "EDA_S_BA_RemoteAccess", 
        "_Pending", "EDA_S_BA_Internetzugriff",
        "EDA_ANW_DMS Fabasoft eGov Suite",
        "EDA_S_APS"
        "EDA_S_BA_ServerAusland",
        "EDA_ANW_ARS Remedy"
        "EDA_S_Büroautomation",
        "EDA_ANW_MOVE!",
        "EDA_ANW_Internet EDA",
        "EDA_ANW_at Honorarvertretung",
        "EDA_S_Netzwerk Inland",
        "EDA_ANW_Plato-HH",
        "EDA_ANW_Zentrale Raumreservation EDA",
        "EDA_ANW_IAM Tool EDA",
        "EDA_ANW_Office Manager",
        "EDA_ANW_ITDoc Sharepoint",
        "EDA_S_Backup & Restore",
        "EDA_ANW_FDFA Security App",
        "EDA_ANW_NOS:4",
        "EDA_ANW_ORBIS",
        "EDA_ANW_EDAContacts",
        "EDA_ANW_EDA PWC Tool",
        "EDA_ANW_EDAssist+",
        "EDA_ANW_CodX PostOffice",
        "EDA_ANW_eVERA",
        "EDA_ANW_Reisehinweise",
        "EDA_ANW_ARIS (EDA Scout)",
        "EDA_ANW_Zeiterfassung SAP",
        "EDA_S_Arbeitsplatzdrucker"]
        
        df = df[df['ServiceProcessed'].map(lambda x: not(x in infrequent))]
        labels = list(set([x for x in df['ServiceProcessed']]))
        #Train and test split with 80:20 ratio 
        #df['ServiceProcessed'] = df['ServiceProcessed'].apply(lambda x: others_name if not(x in labels) else x)
        #df.loc[df['ServiceProcessed'] not in labels] = others_name 
        To_Process = df[["Combined_Content", "ServiceProcessed"]]
        train, test = train_test_split(To_Process, test_size=0.2,random_state=42) #prev: 0.2 but unbalanced dataset

        # Define the sequence lengths, max number of words and embedding dimensions # Sequence length of each sentence. If more, truncate. If less, pad with zeros

        MAX_SEQUENCE_LENGTH = 64
        # Top 20000 frequently occurring words 

        MAX_NB_WORDS = NBs_list[0]#iter_NB_WORDS#2000

        # Get the frequently occurring words 
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS) 
        tokenizer.fit_on_texts(train.Combined_Content) 
        print(test.head())
        train_sequences = tokenizer.texts_to_sequences(train["Combined_Content"]) 
        test_sequences = tokenizer.texts_to_sequences(test["Combined_Content"])

        # dictionary containing words and their index 
        word_index = tokenizer.word_index 
        # print(tokenizer.word_index) 
        # total words in the corpus 
        print('Found %s unique tokens.' % len(word_index))

        # get only the top frequent words on train 

        train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        # get only the top frequent words on test 
        test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)


        print(train_data.shape) 
        print(test_data.shape)

        train_labels = train['ServiceProcessed'] 
        test_labels = test['ServiceProcessed']


        from sklearn.preprocessing import LabelEncoder
        # converts the character array to numeric array. Assigns levels to unique labels.

        le = LabelEncoder() 
        le.fit(train_labels) 
        train_labels = le.transform(train_labels) 
        test_labels = le.transform(test_labels)

        print(le.classes_) 
        print(np.unique(train_labels, return_counts=True)) 
        print(np.unique(test_labels, return_counts=True))


        # changing data types 
        labels_train = to_categorical(np.asarray(train_labels)) 
        labels_test = to_categorical(np.asarray(test_labels)) 
        print('Shape of data tensor:', train_data.shape) 
        print('Shape of label tensor:', labels_train.shape) 
        print('Shape of label tensor:', labels_test.shape)

        EMBEDDING_DIM = iter_DIM# 100
        print(MAX_SEQUENCE_LENGTH)

        print('Training CNN 1D model.')

        model = Sequential() 
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)) 
        model.add(Dropout(0.5)) 
        model.add(Conv1D(70, 5, activation='relu')) 
        model.add(MaxPooling1D(5)) 
        model.add(Dropout(0.5)) 
        model.add(BatchNormalization()) 
        '''
            model.add(Conv1D(224, 5, activation='relu')) 
            model.add(MaxPooling1D(5)) 
            model.add(Dropout(0.5)) 
            model.add(BatchNormalization())
        '''
        model.add(Flatten()) 
        model.add(Dense(30, activation='relu')) 
        model.add(Dropout(0.5)) 
        model.add(BatchNormalization()) 
        model.add(Dense(26, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        model.fit(train_data, labels_train, batch_size=45, epochs=40, validation_data=(test_data, labels_test))
        #predictions on test data

        predicted=model.predict(test_data)
        
        print("***PREDICTED::",predicted)
        print("***labels_test::",labels_test)
        #model evaluation
        #import tensorflow
        #all_labels = labels
        #all_labels.append(others_name)
        #cm_plot_labels = sorted(all_labels)#tensorflow.keras.utils.probas_to_classes(predicted)
        import sklearn 
        from sklearn.metrics import precision_recall_fscore_support as score
        #cm = confusion_matrix(y_true=labels_test, y_pred=predicted.round())
        #plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

        precision, recall, fscore, support = score(labels_test, predicted.round())
        print("f1 SCORE ______ ", f1_m(labels_test, predicted))
        print('precision: {}'.format(precision)) 
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))

        print("############################")

        print(sklearn.metrics.classification_report(labels_test, predicted.round()))
        lens = []
        df['Combined_Content'].apply(lambda x: lens.append(len(x.split(' '))))
        #print(lens)
        print("Average_length: ",sum(lens)/len(lens))
        if model_train:
            pass

        ## Model Training

        ## Model Evaluation

        ## Model Results
    if len(NBs_list) > 1:
        print(NBs_list, NBs_acc)
        plt.plot(NBs_list, NBs_acc)
        plt.xlabel('Number of frequent words used')
        plt.ylabel('F1-Score')
        plt.show()
    if len(DIMs_list) > 1:
        print(DIMs_list, DIMs_acc)
        plt.plot(DIMs_list, DIMs_acc)
        plt.xlabel('Embedding Dimensions')
        plt.ylabel('F1-Score')
        plt.show()
    if model_evaluate:
        cleanedDataPath = "/Users/luca/Offline/data/test_reduced_clean.csv"
        df = pd.read_csv(cleanedDataPath, sep=",", encoding="utf8")
        sequences = tokenizer.texts_to_sequences(df["Combined_Content"]) 
        # dictionary containing words and their index 
        word_index = tokenizer.word_index 
        # print(tokenizer.word_index) 
        # total words in the corpus 
        #print('Found %s unique tokens.' % len(word_index))
        # get only the top frequent words on train 

        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        predicted=model.predict_classes(data)
        print("SORTED::::",sorted(labels))
        print("LABELS:::" , predicted)
        df['Predicted'] = [sorted(labels)[x] for x in predicted]
        df[['Id', 'Predicted']].to_csv("/Users/luca/Offline/data/test_reduced_predict_service-excl-minorities_nowfin2.csv", index=False)


if model_train3:
    import sys, os, re, csv, codecs
    import numpy as np
    import pandas as pd
    import keras
    from keras.preprocessing.text import Tokenizer 
    from keras.preprocessing.sequence import pad_sequences 
    from keras.utils import to_categorical 
    from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation 
    from keras.layers import Bidirectional, GlobalMaxPool1D, Conv1D, SimpleRNN 
    from keras.models import Model 
    from keras.models import Sequential 
    from keras import initializers, regularizers, constraints, optimizers, layers 
    from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization 
    from keras.layers import Conv1D, MaxPooling1D, Embedding
    from keras.models import Sequential
    from sklearn.metrics import confusion_matrix
    import itertools

    import matplotlib.pyplot as plt 
    from sklearn.model_selection import train_test_split
    '''labels = ["EDA_ANW_Intranet/Collaboration EDA", 
    "EDA_ANW_SAP Services",
    "EDA_ANW_SysPeDoc",
    "EDA_S_APS_Monitor",
    "EDA_S_APS_OS_BasisSW",
    "EDA_S_APS_PC",
    "EDA_S_APS_Peripherie",
    "EDA_S_BA_2FA",
    "EDA_S_BA_Account",
    "EDA_S_BA_Mailbox",
    "EDA_S_BA_UCC_Benutzertelefonie",
    "EDA_S_BA_UCC_IVR",
    "EDA_S_Benutzerunterstützung",
    "EDA_S_Betrieb Übermittlungssysteme",
    "EDA_S_Mobile Kommunikation",
    "EDA_S_Netzdrucker",
    "EDA_S_Order Management",
    "EDA_S_Peripheriegeräte",
    "EDA_S_Zusätzliche Software"]'''
    #combined_into_others = ["EDA_S_BA_Datenablage", "EDA_S_BA_Internetzugriff", "EDA_S_BA_RemoteAccess", "EDA_S_IT Sicherheit", "EDA_S_Netzwerk Ausland", "EDA_S_Raumbewirtschaftung"]
    '''others_name = "EDA_others"'''
    def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    NBs_list =[14000] #[int(x) for x in np.linspace(10000,20000,10)]
    DIMs_list = [200]#[int(x) for x in np.linspace(800,1200,5)]
    NBs_acc = []
    DIMs_acc = []
    for iter_DIM in DIMs_list:

        cleanedDataPath = "/Users/luca/Offline/data/clean_expanded_manuals.csv"
        df = pd.read_csv(cleanedDataPath, sep=",", encoding="utf8")
        df = df.reset_index(drop=True)
        #df = df.dropna(subset=['ManualGroups'])
        df['ManualGroups'].replace('','NONE') 
        df = df.sort_values(by='Id', ascending=False)
        df = df.applymap(str)
        #print(df['ServiceProcessed'].value_counts())
        #input()
        #infrequent = []
        '''infrequent = ["EDA_S_Raumbewirtschaftung", 
        "EDA_ANW_ZACWEB", 
        "EDA_S_BA_Datenablage", 
        "EDA_S_BA_RemoteAccess", 
        "_Pending", "EDA_S_BA_Internetzugriff",
        "EDA_ANW_DMS Fabasoft eGov Suite",
        "EDA_S_APS"
        "EDA_S_BA_ServerAusland",
        "EDA_ANW_ARS Remedy"
        "EDA_S_Büroautomation",
        "EDA_ANW_MOVE!",
        "EDA_ANW_Internet EDA",
        "EDA_ANW_at Honorarvertretung",
        "EDA_S_Netzwerk Inland",
        "EDA_ANW_Plato-HH",
        "EDA_ANW_Zentrale Raumreservation EDA",
        "EDA_ANW_IAM Tool EDA",
        "EDA_ANW_Office Manager",
        "EDA_ANW_ITDoc Sharepoint",
        "EDA_S_Backup & Restore",
        "EDA_ANW_FDFA Security App",
        "EDA_ANW_NOS:4",
        "EDA_ANW_ORBIS",
        "EDA_ANW_EDAContacts",
        "EDA_ANW_EDA PWC Tool",
        "EDA_ANW_EDAssist+",
        "EDA_ANW_CodX PostOffice",
        "EDA_ANW_eVERA",
        "EDA_ANW_Reisehinweise",
        "EDA_ANW_ARIS (EDA Scout)",
        "EDA_ANW_Zeiterfassung SAP",
        "EDA_S_Arbeitsplatzdrucker"]'''
        
        #df = df[df['ServiceProcessed'].map(lambda x: not(x in infrequent))]
        labels = list(set([x for x in df['ManualGroups']]))
        #Train and test split with 80:20 ratio 
        #df['ServiceProcessed'] = df['ServiceProcessed'].apply(lambda x: others_name if not(x in labels) else x)
        #df.loc[df['ServiceProcessed'] not in labels] = others_name 
        To_Process = df[["Combined_Content", "ManualGroups"]]
        train, test = train_test_split(To_Process, test_size=0.05,random_state=42) #prev: 0.2 but unbalanced dataset

        # Define the sequence lengths, max number of words and embedding dimensions # Sequence length of each sentence. If more, truncate. If less, pad with zeros

        MAX_SEQUENCE_LENGTH = 64
        # Top 20000 frequently occurring words 

        MAX_NB_WORDS = NBs_list[0]#iter_NB_WORDS#2000

        # Get the frequently occurring words 
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS) 
        tokenizer.fit_on_texts(train.Combined_Content) 
        print(test.head())
        train_sequences = tokenizer.texts_to_sequences(train["Combined_Content"]) 
        test_sequences = tokenizer.texts_to_sequences(test["Combined_Content"])

        # dictionary containing words and their index 
        word_index = tokenizer.word_index 
        # print(tokenizer.word_index) 
        # total words in the corpus 
        print('Found %s unique tokens.' % len(word_index))

        # get only the top frequent words on train 

        train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        # get only the top frequent words on test 
        test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)


        print(train_data.shape) 
        print(test_data.shape)

        train_labels = train['ManualGroups'] 
        test_labels = test['ManualGroups']


        from sklearn.preprocessing import LabelEncoder
        # converts the character array to numeric array. Assigns levels to unique labels.

        le = LabelEncoder() 
        le.fit(train_labels) 
        train_labels = le.transform(train_labels) 
        test_labels = le.transform(test_labels)

        print(le.classes_) 
        print(np.unique(train_labels, return_counts=True)) 
        print(np.unique(test_labels, return_counts=True))


        # changing data types 
        labels_train = to_categorical(np.asarray(train_labels)) 
        labels_test = to_categorical(np.asarray(test_labels)) 
        print('Shape of data tensor:', train_data.shape) 
        print('Shape of label tensor:', labels_train.shape) 
        print('Shape of label tensor:', labels_test.shape)

        EMBEDDING_DIM = iter_DIM# 100
        print(MAX_SEQUENCE_LENGTH)

        print('Training CNN 1D model.')

        model = Sequential() 
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)) 
        model.add(Dropout(0.5)) 
        model.add(Conv1D(70, 5, activation='relu')) 
        model.add(MaxPooling1D(5)) 
        model.add(Dropout(0.5)) 
        model.add(BatchNormalization()) 
        '''
            model.add(Conv1D(224, 5, activation='relu')) 
            model.add(MaxPooling1D(5)) 
            model.add(Dropout(0.5)) 
            model.add(BatchNormalization())
        '''
        model.add(Flatten()) 
        model.add(Dense(30, activation='relu')) 
        model.add(Dropout(0.5)) 
        model.add(BatchNormalization()) 
        model.add(Dense(len(labels), activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        model.fit(train_data, labels_train, batch_size=45, epochs=35, validation_data=(test_data, labels_test))
        #predictions on test data

        predicted=model.predict(test_data)
        
        print("***PREDICTED::",predicted)
        print("***labels_test::",labels_test)
        #model evaluation
        #import tensorflow
        #all_labels = labels
        #all_labels.append(others_name)
        #cm_plot_labels = sorted(all_labels)#tensorflow.keras.utils.probas_to_classes(predicted)
        import sklearn 
        from sklearn.metrics import precision_recall_fscore_support as score
        #cm = confusion_matrix(y_true=labels_test, y_pred=predicted.round())
        #plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

        precision, recall, fscore, support = score(labels_test, predicted.round())
        print("f1 SCORE ______ ", f1_m(labels_test, predicted))
        print('precision: {}'.format(precision)) 
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))

        print("############################")

        print(sklearn.metrics.classification_report(labels_test, predicted.round()))
        lens = []
        df['Combined_Content'].apply(lambda x: lens.append(len(x.split(' '))))
        #print(lens)
        print("Average_length: ",sum(lens)/len(lens))
        if model_train:
            pass

        ## Model Training

        ## Model Evaluation

        ## Model Results
    if len(NBs_list) > 1:
        print(NBs_list, NBs_acc)
        plt.plot(NBs_list, NBs_acc)
        plt.xlabel('Number of frequent words used')
        plt.ylabel('F1-Score')
        plt.show()
    if len(DIMs_list) > 1:
        print(DIMs_list, DIMs_acc)
        plt.plot(DIMs_list, DIMs_acc)
        plt.xlabel('Embedding Dimensions')
        plt.ylabel('F1-Score')
        plt.show()
    if model_evaluate3:
        cleanedDataPath = "/Users/luca/Offline/data/test_reduced_clean.csv"
        df = pd.read_csv(cleanedDataPath, sep=",", encoding="utf8")
        sequences = tokenizer.texts_to_sequences(df["Combined_Content"]) 
        # dictionary containing words and their index 
        word_index = tokenizer.word_index 
        # print(tokenizer.word_index) 
        # total words in the corpus 
        #print('Found %s unique tokens.' % len(word_index))
        # get only the top frequent words on train 

        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        predicted=model.predict(data)
        fins = []
        sortlabs = sorted(labels)
        for l in predicted:
            jj = l.argsort()[-3:][::-1]
            labs = []
            for x in jj:
                labs.append(sortlabs[x])
            if labs[0] == "NONE" or not(labs[0] in labels):
                fins.append('')
            else:
                fins.append(labs[0]+'|'+labs[1]+'|'+labs[2])
        #fins = [x if not('nan' in x) else '' for x in fins]
        print('PREDICTED:', fins)
        print("SORTED::::",sorted(labels))
        print("LABELS:::" , predicted)
        df['Predicted'] = fins
        df[['Id', 'Predicted']].to_csv("/Users/luca/Offline/data/test_reduced_predict_MANUAL.csv", index=False)