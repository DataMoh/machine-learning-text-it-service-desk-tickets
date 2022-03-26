
## Bert Classifier with Undersampling


import pandas as pd
import numpy  as np
from  time import time
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score,precision_score, recall_score, f1_score
import seaborn as sn

import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_text as tftext
from tensorflow.keras import metrics


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
def undersample_shuffle(x, y, random_state=42):
    
    "oversamples x and y for equal class proportions"
    print('Original dataset shape %s' % Counter(y))
    
    #split data in to 70/30 split - need to check if this split is optimal 
    text_train, text_test, target_train, target_test = \
    train_test_split(x, y, train_size=0.8, random_state=12345, shuffle=True, stratify=y)
    
    rus = RandomUnderSampler(random_state=42)
    
    
    x_train1, y_train1 = rus.fit_resample(text_train, target_train)
    x_test1, y_test1    = rus.fit_resample(text_test, target_test)
    print('Resampled train dataset shape %s' % Counter(y_train1))
    print('Resampled test dataset shape %s' % Counter(y_test1))
    return x_train1, y_train1, x_test1, y_test1




#consolidate text and target data to one dateframe for oversampling 
x_train1, y_train1, x_test1, y_test1 = undersample_shuffle(df, target)

#set text data to the train and test 
x_train1 = x_train1['body']
x_test1  = x_test1['body']

yt_dis = Counter(y_train1)
yv_dis = Counter(y_test1)

##BERT Model

bert_preprocess = tfhub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = tfhub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2")
# Layers = 6 Hidden Size (H) = 128 A = 2/2

#Encoder below is too demanding on system. Leads to long processing times. 
# bert_encoder = tfhub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
# Layers = 12 Hidden Size (H) = 768 A = 12/4

##build BERT Classifier
def build_classifer_model():
    
    #Bert Layers
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    #TensorFlow NN Layers
    activation = 'softmax' # softmax is prefered for multiclassificaiton over sigmoid
    l = tf.keras.layers.Dropout(0.1, name='dropout')(outputs['pooled_output'])
    l = tf.keras.layers.Dense(enc_ytrain1.shape[1], activation=activation, name='output')(l)
    return tf.keras.Model(inputs=[text_input], outputs=[l])

#prepare data for bert model 
enc_ytrain1 = pd.get_dummies(y_train1)
enc_ytest1  = pd.get_dummies(y_test1)

classifier_model = build_classifer_model()
classifier_model.summary()

bert_metrics = [tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")]
bert_loss = 'categorical_crossentropy'

epochs=50

classifier_model.compile(optimizer='adam', loss=bert_loss, metrics = bert_metrics)

history = classifier_model.fit(x_train1, enc_ytrain1, validation_data=(x_test1,enc_ytest1), epochs=epochs)
loss, accuracy = classifier_model.evaluate(x_test1, enc_ytest1)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

pred = classifier_model.predict(x_test1)
class_names = np.unique(y_test1)
pred_f = pd.DataFrame(class_names[np.argmax(pred, axis=1)])

cr = classification_report(y_test1,pred_f)
print(cr)
cm = confusion_matrix(y_test1,pred_f)

#plt.subplot(3, 1, 1)
sn.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y_test1),yticklabels=np.unique(y_test1))
plt.xlabel('Predicted')
plt.ylabel('Truth')

history_dict = history.history
print(history_dict.keys())
acc = history_dict['categorical_accuracy']
val_acc = history_dict['val_categorical_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc)+1)
fig = plt.figure(figsize=(10,6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# r is for "solid red line"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


