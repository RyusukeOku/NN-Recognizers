import torch
import pprint

file_path = 'main.vocab'
data = torch.load(file_path)

pprint.pprint(data)

tokens = data.get('tokens', [])
if '<bos>' in tokens:
    print("Vocabulary contains <bos> token.")
elif 'bos' in tokens:
    print("Vocabulary contains 'bos' token.")
else:
    print("Vocabulary does not contain <bos> or 'bos' token.")