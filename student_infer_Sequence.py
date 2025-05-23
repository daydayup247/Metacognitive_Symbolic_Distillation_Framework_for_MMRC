import os
import tqdm
from datasets import load_dataset
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import csv


model = AutoModelForSeq2SeqLM.from_pretrained('./t5-ft-or')
tokenizer = AutoTokenizer.from_pretrained('./t5-ft-or')
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


def generate_answer(test_samples, model):
    inputs = tokenizer(
        test_samples,
        padding="max_length",
        truncation=True,
        # max_length=max_input_length,
        max_length=1024,
        return_tensors="pt",

    )
    # print('inputs', inputs)
    # inputs['input_ids'] = torch.cat([torch.full((1, n_tokens), 50256), inputs['input_ids']], 1)
    # inputs['attention_mask'] = torch.cat([torch.full((1, n_tokens), 1), inputs['attention_mask']], 1)

    model = model.to(device)
    input_ids = inputs.input_ids.to(model.device)

    attention_mask = inputs.attention_mask.to(model.device)

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1024)
                             # pad_token_id=tokenizer.pad_token_id, early_stopping=True)
    # print('outputs: ', outputs)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


dataset = load_dataset('csv', data_files='../../RACE_JS/high/test.csv')
context = dataset['train']['context']
question = dataset['train']['question']
options = dataset['train']['options']

with open('results/result_or.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['context', 'question', 'options', 'answer'])

    num = len(context)
    # num = 5
    for n in tqdm.tqdm(range(num)):
        _, answer = generate_answer(str(context[n]) + str(question[n]) + str(options[n]), model)
        writer.writerow([context[n], question[n], options[n], answer[0][0]])











