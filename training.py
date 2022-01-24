import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import pickle
import json

lemmatizer = WordNetLemmatizer()


intents = json.loads(open('database/intents.json').read())
words=[]
classes=[]
documents=[]
ignore_letters=['?','!','.',',']

# store words in list
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# lemmatize every word except for ignored words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(classes)

# dump words and classes
pickle.dump(words, open('database/words_medical.pkl', 'wb'))
pickle.dump(classes, open('database/classes_medical.pkl', 'wb'))

# create training set
training=[]
output_empty = [0] * len(classes)

for doc in documents:
    bag=[]
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag,output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

# MODEL
model = Sequential([
    Dense(128,input_shape=(len(train_x[0]),),activation='relu'),
    Dropout(0.5),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]),activation='softmax')
])
print("model defined...")

model.compile(optimizer=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("model compiled...")

hist =  model.fit(np.array(train_x),
                  np.array(train_y),
                  epochs=300,
                  batch_size=5,
                  verbose=1)
print("model trained...")

model.save('chatbot_model.h5',hist)
print("Done. Saved model...")






