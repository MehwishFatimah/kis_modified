import torch
import os
import argparse
import pandas as pd

from nltk.tokenize import word_tokenize, sent_tokenize
from sacremoses import MosesDetokenizer
mt = MosesDetokenizer(lang='en')
from model_generator import Generator

'''-----------------------------------------------------------------------
'''
parser = argparse.ArgumentParser(description = 'Help for the main module')
parser.add_argument('-d', type = str, default = None,   help = 'Data dir')

model_card = "gpt2-medium"
model_file = "/hits/basement/nlp/fatimamh/inputs/kis/models/gpt2_med_keep_it_simple.bin"

max_output_length=200
num_runs=10

'''-----------------------------------------------------------------------
'''
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))
model = Generator(model_card, max_output_length=max_output_length, device=device)
if len(model_file) > 0:
    model.reload(model_file)
model.eval()

'''-----------------------------------------------------------------------
'''
def detokenize_text(text):
    words = text.split(" ")
    #print(words)
    #print()
    text = mt.detokenize(words) 
    #print(text)
    return text

'''-----------------------------------------------------------------------
'''
def find_min_prob(model_output):
    min_prob = -500
    final = ""
    for out in model_output:
        #print('prob: {}'.format(out.get('logprob')))
        if out.get('logprob') > min_prob:
            min_prob = out.get('logprob')
            final = out.get('output_text')
            #print('------------')
            #print('min_prob: {}'.format(min_prob))
            #print('final: : {}'.format(final))
            #print('------------')
            
    return final

'''-----------------------------------------------------------------------
'''
def process_text(text):
    text = text.lower()
    text = word_tokenize(text)
    text = " ".join(text)
    #text = sent_tokenize(text)
    # do it later after handling cross-lingual 
    #print(text)

    return text

'''-----------------------------------------------------------------------
'''
def get_simple_text(text):

    paragraph = detokenize_text(text)
    model_output = model.generate([paragraph], num_runs=num_runs, sample=True)[0]
    final = find_min_prob(model_output)

    final = process_text(final)
    #print("\n\nTEXT: {}\n".format(paragraph))
    
    return final

'''-----------------------------------------------------------------------
'''
def get_string(text):

    #print('-----------------------\ntext: {}\n'.format(text))
    text = text.replace("[\'","")
    text = text.replace("\']", "")
    text = text.replace("\', \'", " ")
    text = text.replace("\'\'\", \'", "")
    text = text.replace("\', \"", " ")
    text = text.replace("\", \'", " ")
    text = text.replace("`` ", "")
    text = text.replace("\'\' ", "")
    text = text.replace("\", \"", " ")
    text = text.replace("[\"","")
    text = text.replace("\"]", "")

    #print('text: {}\n-----------------------\n'.format(text))
    
    return text
'''-----------------------------------------------------------------------
'''

if __name__ == "__main__":

    args = parser.parse_args()
    path = args.d
    file = os.path.join(path, 'summaries.csv')
    print('file: {}'.format(file))
    
    df = pd.read_csv(file, index_col= False) 
    #print(df.head())
    df.drop('Unnamed: 0', axis= 1, inplace=True)
    df.drop('meta', axis= 1, inplace=True)
    #df = df.head(5)

    df['system'] = df['system'].apply(lambda x: get_string(x))
    '''
    for index, row in df.iterrows():
        print ('-----------------------\nindex: {}\n'.format(index))
        print('system: {}\n-----------------------\n'.format(row['system']))
    '''
    df['system'] = df['system'].apply(lambda x: get_simple_text(x))
    '''
    for index, row in df.iterrows():
        print ('-----------------------\nindex: {}\n'.format(index))
        print('system: {}\n-----------------------\n'.format(row['system']))
    '''
    file = os.path.join(path, 'simple_summaries.csv')
    print('file: {}'.format(file))

    df.to_csv(file, index=False)

    

        
    