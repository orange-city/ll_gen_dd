import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.utils.rnn import pad_sequence
from preprocessPTBIO import readData, preProcess, readDataFromConll

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
device_ids = [0, 1, 2]  # List the GPU device IDs you want to use

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model = nn.DataParallel(model, device_ids=device_ids)
model.to(device_ids[0])

# Disfluency sentences data in the given format
disfluency_data = preProcess(readData('dps/swbd/train_2K'))

# Convert Disfluency data to BERT format
def convert_to_bert_format(sentence_data, tokenizer, max_length):
    bert_input = []
    for sentence in sentence_data:
        words = [word[0] for word in sentence]
        input_ids = tokenizer.encode(" ".join(words), add_special_tokens=True, max_length=max_length, truncation=True,truncation_strategy='longest_first')
        bert_input.append(torch.tensor(input_ids))
    padded_input = pad_sequence(bert_input, batch_first=True).to(device_ids[0])
    return padded_input

# Predict quality using BERT
def predict_quality(sentence_data, model):
    model.eval()
    with torch.no_grad():
        # Determine the maximum length in the batch
        max_length = max(len(sentence) for sentence in sentence_data)
        inputs = convert_to_bert_format(sentence_data, tokenizer, max_length)
        outputs = model(inputs)
        logits = outputs[0]  # Access the logits from the tuple
        probabilities = torch.softmax(logits, dim=1)
    return probabilities[:, 1].tolist()  # Probability of being a higher-quality sentence

# Get quality scores for Disfluency sentences
quality_scores = []
batch_size = 8  # Adjust the batch size as needed
for i in range(0, len(disfluency_data), batch_size):
    batch_data = disfluency_data[i:i + batch_size]
    scores = predict_quality(batch_data, model)
    quality_scores.extend(scores)

# Define a threshold to filter out lower-quality sentences
threshold = 0.5

# Select higher-quality Disfluency sentences
higher_quality_disfluency = [sentence_data for sentence_data, score in zip(disfluency_data, quality_scores) if score >= threshold]
sum = 0
num = 0
# Print the selected higher-quality Disfluency sentences
for sentence_data in higher_quality_disfluency:
    #print(sentence_data)
    num += 1
    sum += len(sentence_data)
print(sum)
print(num)