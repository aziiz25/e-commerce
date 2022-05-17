import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from sklearn import svm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack 
from nltk.stem import WordNetLemmatizer


df = pd.read_csv('train_data.csv')
df.sample(10)
df.info()
df.describe()


df['sentiment'].value_counts().plot(kind = 'bar')
plt.show()

df[df.duplicated()].count()

df.drop_duplicates(inplace = True)
df.info()


df['brand'].value_counts()
df['categories'].value_counts()
df['primaryCategories'].value_counts()
df['reviews.date']
df['reviews.title']

#based on the above i see that some columns will not be needed 

df_updated = df[['reviews.text', 'sentiment']].copy()
df_updated.info()
df_updated.sample(10)




df_updated['reviews.text'] = df_updated['reviews.text'].apply(str.lower)
df_updated['reviews.text'] = df_updated['reviews.text'].str.replace('[^a-zA-Z\s]', ' ',regex=True) 

df_updated.sample(10)

df_updated.columns = ['text', 'sentiment']

tweet_tokenizer = TweetTokenizer()

df_updated['text'] = [tweet_tokenizer.tokenize(text) for text in df_updated['text']]
df_updated.sample(10)


lem = WordNetLemmatizer()
df_updated['text'] = [[lem.lemmatize(word) for word in text ] for text in df_updated['text']]
df_updated.sample(10)



stop_words = stopwords.words('english')

df_updated['text'] = [[word for word in text  if word not in stop_words] for text in df_updated['text']]
df_updated.sample(10)


df_updated['updated_text'] = [' '.join(text) for text in df_updated['text']]
df_updated.sample(10)

df_updated['review_length'] = [len(review) for review in df_updated['updated_text']]
df_updated.sample(10)


df_updated['sentiment'] = df_updated['sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive':2})

df_updated.sample(10)


X = df_updated[['updated_text']]
y = df_updated['sentiment']

X.sample(10)
y.sample(10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

text_vect = TfidfVectorizer()
text_vect.fit(X_train['updated_text'])
X_train_transform = text_vect.transform(X_train['updated_text'])
X_train_transform.toarray()
X_test_transform = text_vect.transform(X_test['updated_text'])


def model_training(model, X_train = X_train_transform, y_train = y_train, X_test = X_test_transform):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def model_score(y_pred):   
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


model_score(model_training(MultinomialNB()))
model_score(model_training(RandomForestClassifier(max_depth=40, n_estimators= 30, class_weight= 'balanced')))

params = {
    'objective': 'multi:softmax',
    'max_depth': 5,
    'learning_rate': 0.01,
    'silent': True,
    'n_estimators': 500,
    'early_stopping_round':10,
    'num_class':3
}

model_score(model_training(XGBClassifier(**params)))
model_score(model_training(svm.SVC()))
from sklearn.tree import DecisionTreeClassifier
model_score(model_training(DecisionTreeClassifier(class_weight='balanced')))
from sklearn.linear_model import LogisticRegression
model_score(model_training(LogisticRegression(solver='saga', 
                        max_iter=5000, 
                        multi_class='ovr', 
                        class_weight= 'balanced')))


#---------------------------------------------------------------------------------------------------------

analyzer = SentimentIntensityAnalyzer()

df_updated['score']= [analyzer.polarity_scores(text)['compound'] for text in df_updated['updated_text']]

X = df_updated[['updated_text', 'score']]
y = df_updated['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


text_vect = TfidfVectorizer()
text_vect.fit(X_train['updated_text'])
X_train_transform = text_vect.transform(X_train['updated_text'])
X_train_transform.toarray()
X_test_transform = text_vect.transform(X_test['updated_text'])

X_train_transform = hstack([X_train_transform, X_train['score'].values.reshape(-1, 1)])
X_test_transform = hstack([X_test_transform, X_test['score'].values.reshape(-1, 1)])

X_train_transform


model_score(model_training(MultinomialNB(), X_train_transform,y_train ,X_test_transform))
model_score(model_training(RandomForestClassifier(max_depth=40, n_estimators= 30, class_weight= 'balanced'), X_train_transform,y_train ,X_test_transform))

params = {
    'objective': 'multi:softmax',
    'max_depth': 5,
    'learning_rate': 0.01,
    'silent': True,
    'n_estimators': 500,
    'early_stopping_round':10,
    'num_class':3
}

model_score(model_training(XGBClassifier(**params), X_train_transform,y_train ,X_test_transform))
model_score(model_training(svm.SVC(), X_train_transform,y_train ,X_test_transform))


#-------------------------------

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
np.unique(y_train, return_counts=True)

X_res, y_res = ros.fit_resample(X_train_transform, y_train)
model_score(model_training(RandomForestClassifier(max_depth=40, n_estimators= 30, class_weight= 'balanced'), X_res, y_res, X_test_transform))
model_score(model_training(XGBClassifier(**params),X_res, y_res, X_test_transform))
model_score(model_training(svm.SVC(),X_res, y_res, X_test_transform))
#------------------------------------------------------------------------------
X = df_updated[['updated_text']]
y = df_updated['sentiment']

X.sample(10)
y.sample(10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

import nlpaug.augmenter.word as naw


aug_w2v = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert", device = 'cuda')

def augment_text(train,samples=1000,target = 0):
    df_n=train[train['sentiment'] == target].reset_index(drop=True)
    new_text = []
    for i in np.random.randint(0,len(df_n),samples):        
        text = df_n.iloc[i]['updated_text']
        augmented_text = aug_w2v.augment(text)
        new_text.append(augmented_text)
    new=pd.DataFrame({'updated_text':new_text,'sentiment':target})
    train = train.append(new).reset_index(drop=True)
    return train

df_test = X_train.copy()
df_test['sentiment'] = y_train
df_test.sample(10)
df_test = augment_text(df_test,samples=300 ,target = 1)
df_test = augment_text(df_test,samples=300 ,target = 0)
df_test['sentiment'].value_counts()
X_train_aug = df_test['updated_text']
y_train_aug = df_test['sentiment']
X_train_transform_aug = text_vect.transform(X_train_aug)

model_score(model_training(RandomForestClassifier(class_weight= 'balanced'), X_train_transform_aug, y_train_aug, X_test_transform))
model_score(model_training(XGBClassifier(**params), X_train_transform_aug,y_train_aug ,X_test_transform))
model_score(model_training(MultinomialNB(), X_train_transform_aug,y_train_aug ,X_test_transform))

df_test['score']= [analyzer.polarity_scores(text)['compound'] for text in df_test['updated_text']]
df_test[df_test['sentiment'] == 0].sample(10)

X_train_transform_aug = hstack([X_train_transform_aug, df_test['score'].values.reshape(-1, 1)])
X_test_transform = hstack([X_test_transform, X_test['score'].values.reshape(-1, 1)])

model_score(model_training(RandomForestClassifier(), X_train_transform_aug, y_train_aug, X_test_transform))
model_score(model_training(XGBClassifier(), X_train_transform_aug,y_train_aug ,X_test_transform))


model_score(model_training(MultinomialNB(), X_train_transform_aug,y_train_aug ,X_test_transform))

from sklearn.tree import DecisionTreeClassifier
model_score(model_training(DecisionTreeClassifier(class_weight='balanced')))
from sklearn.linear_model import LogisticRegression
model_score(model_training(LogisticRegression(solver='lbfgs', 
                        max_iter=10000, 
                        multi_class='ovr', 
                        class_weight= 'balanced'),X_train_transform_aug,y_train_aug ,X_test_transform))

#------------------------------------------------------------------------------

from tensorflow.keras.layers import Embedding,LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras import utils


y_train = utils.to_categorical(y_train_aug)
y_test = utils.to_categorical(y_test)

y_train.shape


### Vocabulary size
voc_size=10000

onehot_train=[one_hot(words,voc_size)for words in X_train_aug] 
print(onehot_train[:3])

sent_length=df_updated['review_length'].max()
embedded_docs=pad_sequences(onehot_train,padding='pre',maxlen=sent_length)
print(embedded_docs[:3])

onehot_test = [one_hot(words,voc_size) for words in X_test['updated_text']]
embedded_docs_test =pad_sequences(onehot_test,padding='pre',maxlen=sent_length)
print(embedded_docs[:3])

dim = 40


model = Sequential()
model.add(Embedding(voc_size, dim, input_length= sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

X_train_final = np.array(embedded_docs)
X_test_final = np.array(embedded_docs_test)

history = model.fit(X_train_final, y_train, validation_data = (X_test_final, y_test) , epochs = 3 , batch_size = 64)


#Confution Matrix and Classification Report
Y_pred = model.predict(X_test_final)
y_pred = np.argmax(Y_pred, axis=1)
y_test_result = np.argmax(y_test, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_test_result, y_pred))
print('Classification Report')
target_names = [str(i) for i in range(0,3)]
print(classification_report(y_test_result, y_pred, target_names=target_names))

print(history.history.keys())

with plt.style.context('dark_background'):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()




import seaborn as sns

def model_detail():
    with plt.style.context('dark_background'):
        cm = confusion_matrix(y_test_result, y_pred)
        f = sns.heatmap(cm, annot=True, fmt='d')
        f.set_title("confusion_matrix" , color = "white")
        plt.xlabel("Predicted label " , color = "white")
        plt.ylabel("True label " , color = "white")
        plt.show()
    print(classification_report(y_test_result,y_pred))


model_detail()











#------------------------------------------------------------------------------------------


#topic model
import gensim
from nltk import pos_tag
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import re
review_pos  = [pos_tag(text) for text in df_updated['text']]

review_pos[:4]

nouns = []
for text in review_pos:
    temp = []
    for noun in text:
        if re.match('N[NP].*', noun[1]):
            temp.append(noun)
    nouns.append(temp)
nouns[:3]

clean_data = []
for noun in nouns:
    temp =[]
    for token in noun:
        if token[0] not in stop_words and len(token[0]) >=4 and token[0].isalpha():
            temp.append(token[0])
    if len(temp) != 0:
        clean_data.append(temp)



id2word = gensim.corpora.Dictionary(clean_data)


corpus = [id2word.doc2bow(text) for text in clean_data]


corpus



def calculate_topic_cv(topic_range):
  cv_score =[]
  topic_num = []
  for i in range(2,topic_range):
    topic_num.append(i)
    ldamodel = LdaModel(corpus = corpus, num_topics= i, id2word= id2word, passes= 10, random_state= 2)
    cv_score.append(CoherenceModel(model=ldamodel,texts=clean_data, dictionary=id2word , coherence='c_v').get_coherence())
    print('topic {i}: {cv}'.format(i = i, cv = cv_score[i-2]))
  return topic_num,cv_score

topic_num,cv_score = calculate_topic_cv(13)


num_of_topics = 4

lda = LdaModel(corpus = corpus, num_topics= num_of_topics, id2word= id2word, passes= 10, random_state=2)


print('LDA model')
for idx in range(num_of_topics):
    print('Topic #%s:'%idx , lda.print_topic(idx,12))



coherence_lda_model = CoherenceModel(model= lda,texts= clean_data, dictionary= id2word, coherence = 'c_v')


print('coherence score: ', coherence_lda_model.get_coherence())



pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda, corpus, id2word)

pyLDAvis.save_html(vis, 'lda12.html')
vis


lda.print_topics()


lda_topics= lda.show_topics(formatted=False)

topics_only = []
for topic in lda_topics:
    temp_str  = ''
    for sub_topic in topic[1]:
        temp.append(sub_topic[0])
        temp_str += sub_topic[0] + ', '
    topics_only.append([topic[0] ,temp_str])

topics_only



topics_df = pd.DataFrame(topics_only, columns = ['Topic Number','Topic top words'])
topics_df['Topic name'] = ['Phone features', 'Phone issues', 'battery issues', 'Phone screen and speakers', 'Phone network issues','Mixed']


for sent in lda[corpus]:
  print(sent)


final_review = pd.DataFrame([', '.join(sent) for sent in clean_data], columns = ['Review keywords'])


topic_number = []
for sent in lda[corpus]:
  temp = []
  other = []
  for topic_num in sent:
    if topic_num[1] >= 0.35:
      temp.append(topic_num[0])
  if(len(temp) >= 1):
    topic_number.append(temp)
  else:
    topic_number.append([max(sent,key=itemgetter(1))[0]])
topic_number


final_review['Topic Number'] = [', '.join(map(str,number)) for number in topic_number]
final_review

#final_review.columns = ['Topic name']
topic_names = []
for topic_num in topic_number:
    temp = []
    for i in topic_num:
        temp.append(topics_df.iloc[i]['Topic name'])
    topic_names.append(', '.join(temp))
final_review['Topic name'] = topic_names

final_review

final_review['Topic name'].value_counts()[:6]

with plt.style.context('dark_background'):
    plt.grid(color='w', linestyle='solid') 
    plt.ticklabel_format(useOffset = False,style='plain')
    plt.style.use('classic')
    final_review['Topic name'].value_counts()[:6].plot(
        kind ='bar',figsize=(16,9), xlabel= 'Topics',
        ylabel= 'amount in each topic', color= colors,title='Show top 6 topics')
    plt.show()