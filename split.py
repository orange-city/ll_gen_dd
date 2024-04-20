from preprocessPTBIO import readData,preProcess
def extract_sentences(data_list):
    sentences = []
    for sentence_data in data_list:
        sentence = " ".join(word[2] for word in sentence_data)
        sentences.append(sentence)
    return sentences

data = preProcess(readData('dps/swbd/fake_data'))
#data2= preProcess(readData('dps/swbd/fake_data'))

sentences = extract_sentences(data)
#sentences_2= extract_sentences(data2)
# with open('traintag_real.txt', 'w') as f:
#     for sentence in sentences:
#         f.write(sentence + '\n')
#
with open('traintag_fake7002.txt', 'w') as f:
     for sentence in sentences:
         f.write(sentence + '\n')

