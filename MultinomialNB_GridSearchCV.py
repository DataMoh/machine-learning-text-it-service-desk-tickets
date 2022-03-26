
## Multinomial NB with GridSearchCV


import pandas as pd
import numpy  as np
from  time import time
import matplotlib.pyplot as plt
from collections import Counter
from AdvancedAnalytics.Text import text_analysis, text_plot
from imblearn.over_sampling import RandomOverSampler 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score,precision_score, recall_score, f1_score
import seaborn as sn
from wordcloud import WordCloud, STOPWORDS


import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


filename = "all_tickets_significant_cat.xlsx"
df   = pd.read_excel(filename)      

Counter(df['category'])

text = df['body'] #text to analyze 
target = df.drop(['body','title'], axis=1) #target variables to classify
target_v = "category"
# category
# urgency
# impact
target = target[target_v]


##Sample data to even distribution between classes of the target variable. 
def oversample_shuffle(x, y, random_state=42):
    
    "oversamples x and y for equal class proportions"
    print('Original dataset shape %s' % Counter(y))
    
    #split data in to 70/30 split - need to check if this split is optimal 
    text_train, text_test, target_train, target_test = \
    train_test_split(x, y, train_size=0.8, random_state=12345, shuffle=True, stratify=y)
    
    ros = RandomOverSampler(random_state=42)
    
    
    x_train1, y_train1 = ros.fit_resample(text_train, target_train)
    x_test1, y_test1    = ros.fit_resample(text_test, target_test)
    print('Resampled train dataset shape %s' % Counter(y_train1))
    print('Resampled test dataset shape %s' % Counter(y_test1))
    return x_train1, y_train1, x_test1, y_test1


#consolidate text and target data to one dateframe for oversampling 
x_train1, y_train1, x_test1, y_test1 = oversample_shuffle(df, target)

#set text data to the train and test 
x_train1 = x_train1['body']
x_test1  = x_test1['body']

yt_dis = Counter(y_train1)
yv_dis = Counter(y_test1)

ta   = text_analysis(synonyms=None, stop_words=None, pos=True, stem=True) 

max_features    = None  # default is None
max_df          = 1.0 #0.95  # max proportion of docs/reviews allowed for a term
min_df          =  1 #0.05
analyzer        = ta.analyzer
stop_words      = None

parameters = {'vect__ngram_range':[(1,1),(1,2)],
              'vect__use_idf':(True, False),
              'clf__alpha':(1e-2, 1e-3),
              'clf__fit_prior':(True, False),
              }

text_clf = Pipeline([
        ('vect',TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features,
            binary=False, analyzer=analyzer, stop_words=stop_words)),
        ('clf', MultinomialNB())
])

#terms = text_clf['vect'].get_feature_names_out()
text_clf = text_clf.fit(x_train1, y_train1)

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(x_train1, y_train1)

results = pd.DataFrame(gs_clf.cv_results_)
scoring = results[['param_vect__ngram_range','param_vect__use_idf','param_clf__alpha'\
         ,'param_clf__fit_prior','mean_test_score']]
pred = gs_clf.predict(x_test1)
pred_acc = accuracy_score (y_test1, pred)
print('Model Accuracy:', pred_acc)
cr = classification_report(y_test1,pred)
print(cr)
cm = confusion_matrix(y_test1,pred)
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
