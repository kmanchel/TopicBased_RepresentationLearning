import nltk
import string
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from urllib.parse import urlparse
lemmatizer = WordNetLemmatizer()

pd.set_option('display.max_columns', None)

def content(element):
    return (element.text or '') + ''.join(ET.tostring(e, 'unicode') for e in element)


def iter_data(data):
    tree = ET.iterparse(data)
    for event, element in tqdm(tree):
        if element.tag == "article":
            doc_dict = element.attrib
            content_text = content(element)
            doc_dict['content'] = content_text
            yield doc_dict

def base_cleaner(df,lower_case):
    """"
    This cleaning function just removes remaining XML Tags, urls and punctuation.
    Lowercasing can also be done if lower_case == True.
    """"
    clean_xml_tags = re.compile('<.*?>')
    clean_puncts = re.compile('[^.,a-zA-Z0-9 \n\.]')
    clean_comma = re.compile(',')
    df['content'] = df['content'].apply(lambda x: re.sub(clean_xml_tags, '', x))
    df['content'] = df['content'].apply(lambda x: re.sub(clean_puncts, '', x))
    df['content'] = df['content'].apply(lambda x: re.sub(clean_comma, '', x))
    df['content'] = df['content'].apply(lambda x: re.sub(re.compile('\n'), '', x))
    if lower_case:
        df['content'] = df['content'].apply(lambda x: x.lower())
    
    return df


def iter_truth(data):
    tree = ET.iterparse(data)
    for event, element in tree:
        if element.tag == "article":
            doc_dict = element.attrib
            yield doc_dict


def clean_truth(df, bias_values):
    """"
    Cleans up source URLs
    """"
    for index, row in df.iterrows():
        uri = urlparse(row.url)
        row.url = '{uri.scheme}://{uri.netloc}/'.format(uri=uri)
        
        try:
            if not np.isnan(row.bias) and row.bias not in ['', "", "NaN"]:
                row.bias = bias_values[row.bias]
        except:
            if not pd.isnull(row.bias) and row.bias not in ['', "", "NaN"]:
                row.bias = bias_values[row.bias]
    df.hyperpartisan.replace(['true', 'false'], [1, 0], inplace=True)
    return df

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemma(paragraph):
    """"
    Lemmatizes text 
    """"
    
    sent_tokens = nltk.sent_tokenize(paragraph)
    lemmatized_tokens = []
    for sent in sent_tokens:
        lemmas = " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sent) if w not in string.punctuation])
        lemmatized_tokens.append(lemmas)
    return ". ".join(lemmatized_tokens)
        
def remove_stopwords(paragraph):
    """"
    Removes Stopwords
    """"
    sent_tokens = nltk.sent_tokenize(paragraph)
    cachedStopWords = stopwords.words("english")

    cleaned_tokens = []
    for sent in sent_tokens:
        clean = " ".join([word for word in sent.split() if word.lower() not in cachedStopWords])
        cleaned_tokens.append(clean)
    return ". ".join(cleaned_tokens)
#print(" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(df['content'].iloc[0]) if w not in string.punctuation]))

def extra_cleaner(df):
    """"
    This cleaning function applies lemmatization and stopword removal to the content.
    """"
    
    tqdm.pandas()
    #remove stopwords
    print('removing stopwords')
    df['content'] = df['content'].progress_apply(remove_stopwords)
    #lemmatize
    print('lemmatizing')
    df['content'] = df['content'].progress_apply(lemma)
    
    return df
        
def parse_xml_files(data, truth, save, outfile, lower_case, extra_clean):
    """"
    Main function which takes arguments to convert XML dataset to clean CSV version.
    """"
    columns_data = ['id', 'published-at', 'title', 'content']
    columns_truth = ['id', 'hyperpartisan', 'bias', 'url']
    bias = {'left': -1, 'left-center': -0.5, 'least': 0, 'right-center': 0.5, 'right': 1}
    data_df = pd.DataFrame(columns=columns_data)
    truth_df = pd.DataFrame(columns=columns_truth)
    for file in data:
        with open(file, encoding="utf8") as f:
            print("Starting with data file: " + file)
            df = pd.DataFrame.from_records(list(iter_data(f)))
            data_df = data_df.append(df, sort=True)
            print("Finished data file: " + file)
        data_df.to_csv('data_df.csv')
    if not extra_clean:
        data_df = base_cleaner(data_df, lower_case)
    else:
        print("DOING THE EXTRA CLEAAAAN!")
        data_df = base_cleaner(data_df, True)
        data_df = extra_cleaner(data_df)
    #data_df.to_csv('data_df_fin.csv')
    for file in truth:
        with open(file, encoding="utf8") as f:
            print("Starting with truth file: " + file)
            df = pd.DataFrame.from_records(list(iter_truth(f)), columns=columns_truth)
            truth_df = truth_df.append(df, sort=True)
            print("Finished truth file: " + file)
    truth_df = clean_truth(truth_df, bias)
    #print(data_df.head())
    #print(truth_df.head())
    result = pd.merge(data_df, truth_df, on='id')
    result = result[result['bias'].notna()]
    if save:
        result.to_csv(outfile)