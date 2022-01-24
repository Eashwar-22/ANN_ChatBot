import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('database/intents.json').read())
words = pickle.load(open('database/words.pkl', 'rb'))
classes = pickle.load(open('database/classes.pkl', 'rb'))
model = load_model('database/chatbot_model.h5')

def clean_input(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_input(sentence)
    bag=[0]* len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w:
                bag[i] = 1

    return np.array(bag)

def predict_class(input):
    bow = bag_of_words(input)
    res = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > error_threshold]
    results.sort(key = lambda x:x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],
                            'probability':r[1]})
    return return_list

def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    prob = intents_list[0]['probability']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result = random.choice(i['responses'])
            break
    return result,prob

print("Chatbot is running")

# while True:
#     print("----------------------------------------------------------------")
#     message=input("You : ")
#     ints = predict_class(message)
#     res,prob = get_response(ints,intents)
#     print("Bot : ",res,prob)



