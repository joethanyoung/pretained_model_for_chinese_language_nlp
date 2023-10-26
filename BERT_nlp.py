# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:34:22 2023

@author: joethan.zx
"""


# =============================================================================
# import required machine learning packages
from nltk.translate.bleu_score import SmoothingFunction
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import concurrent.futures
from scipy.spatial.distance import cosine
from typing import List
import openpyxl
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from spacy_pkuseg import pkuseg
import itertools
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:300'
os.environ['HTTP_PROXY'] = '172.19.0.1:8080'
os.environ['HTTPS_PROXY'] = '172.19.0.1:8080'

# =============================================================================
# import dataframe using pandas
df = pd.read_excel(
    r'C:\Users\zhuoxun.yang001\Documents\fude\L06103基础表投诉处理清单（黑龙江）.xlsx')

# check the structure of the dataframe
# check the head, tail, and shape of dataframe
df.head(5), df.tail(5), df.shape

# subset the dataframe
df = df[['投诉事由']]

# check the dataframe
df.head(5)

# access values using index
df['投诉事由'][0]

# ================================================================preprocessing

# Flatten the list of lists into a single list of sentences
sentences = []
for i in df['投诉事由']:
    sentences.append(sent_tokenize(i))

# Tokenize sentences using pkuseg and join the tokens
sentences = [''.join(words) for words in sentences]
for sentence in sentences:
    print(sentence)

# import pkuseg
seg = pkuseg()
sentences = [''.join(seg.cut(sentence)) for sentence in sentences]

# check if pytorch is available
torch.cuda.is_available()

if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")
    # Print GPU details
    print("Device name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
else:
    print("GPU is not available. Using CPU instead.")
    device = torch.device("cpu")

# Load the pre-trained model and tokenizer
model_name = "hfl/chinese-bert-wwm-ext"  # A pre-trained Chinese BERT model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# set up function to extract summary


def extract_summary(sentences, top_k=10, batch_size=32, max_length=512):
    num_batches = len(sentences) // batch_size + \
        (1 if len(sentences) % batch_size != 0 else 0)
    all_scores = []

    for batch_idx in range(num_batches):
        batch_sentences = sentences[batch_idx *
                                    batch_size:(batch_idx + 1) * batch_size]

        # Truncate sentences to max_length tokens
        inputs = tokenizer(batch_sentences, padding=True, truncation=True,
                           return_tensors="pt", max_length=max_length)

        # Move input tensors to GPU if available
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        scores = torch.softmax(logits, dim=1)[:, 1]
        all_scores.extend(scores.tolist())

        # Clear GPU memory
        torch.cuda.empty_cache()

    sorted_indices = np.argsort(all_scores)[::-1]
    top_indices = sorted_indices[:top_k].tolist()
    top_sentences = [sentences[i] for i in top_indices]
    return top_sentences


# Ensure the model is running on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# using gpu to process model
model.to(device)

# extract summary for top 10 sentences
top_sentences = extract_summary(sentences, top_k=10)

for i, sentence in enumerate(top_sentences):
    print(f"Top sentence {i + 1}: {sentence}")

# performance metrics
# define a function to calculate scores of generated summary and reference summaries

# define a function to calculate bleu score
# import smoothingfunction from nltk


def calculate_bleu_score(generated_summary, reference_summary):
    tokenized_generated_summary = generated_summary.split()
    tokenized_reference_summary = reference_summary.split()
    # You can choose different smoothing methods (method1 to method7)
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(
        [tokenized_reference_summary], tokenized_generated_summary, smoothing_function=smoothie)
    return bleu_score


# Combine the top sentences into a single string
generated_summary = ' '.join(top_sentences)

# Combine reference summaries into a list of lists
reference_summaries_list = df

# Calculate BLEU score for each set of reference summaries
# calculate instrinsic metrics
for idx, row in df.iterrows():
    reference_summary = row['投诉事由']
    bleu_score = calculate_bleu_score(generated_summary, reference_summary)
    print(f"BLEU Score for Reference Summary {idx + 1}: {bleu_score}")

# Create an empty list to store BLEU scores
bleu_scores = []

# Assuming your DataFrame is named 'df' and has a column named 'summary'
for idx, row in df.iterrows():
    reference_summary = row['投诉事由']
    bleu_score = calculate_bleu_score(generated_summary, reference_summary)
    bleu_scores.append(bleu_score)

# Get indices of top 10 BLEU scores
top_10_indices = sorted(range(len(bleu_scores)),
                        key=lambda i: bleu_scores[i], reverse=True)[:10]

# Print top 10 BLEU scores
for idx in top_10_indices:
    print(f"BLEU Score for Reference Summary {idx + 1}: {bleu_scores[idx]}")
