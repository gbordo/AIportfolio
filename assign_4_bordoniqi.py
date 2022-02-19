import pandas as pd
import nltk.corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import regex as re
import string
import os



nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

#In summary column 
#perform tokenization 
#perform stemming 
#perform lemmatization

data = pd.read_csv(f'{os.getcwd()}/data/Musical_instruments_reviews.csv')


lemma = WordNetLemmatizer()
stem = SnowballStemmer('english')


stop_words = stopwords.words('english')
punkt = re.compile(f"[{string.punctuation}]") 

def preprocess(sent,stopwords,punkt):
    
    #remove punctuation
    prep_sent = re.sub(punkt,"",sent)
    

    #Remove stopwords and lower case
    prep_sent = ' '.join([ w for w in prep_sent.lower().split() if w not in stopwords])
 
    #Lemmatize 
    prep_sent = ' '.join([lemma.lemmatize(w) for w in prep_sent.split()])
    
    #stem 
    prep_sent = ' '.join([stem.stem(w) for w in prep_sent.split()])
    
    return prep_sent
    

data['clean_summary'] = data['summary'].apply(lambda x: preprocess(x,stop_words,punkt))


#tokenize clean summary 

data['token_clean_summary']= data['clean_summary'].apply(word_tokenize)

#Append all the tokenized words in one list

list_of_clean_tokens = data['token_clean_summary'].to_list()

print(f'There are {len(list_of_clean_tokens)} clean tokens')

print(f'Here are a subset of clean Tokens \n {list_of_clean_tokens[0:10]}')