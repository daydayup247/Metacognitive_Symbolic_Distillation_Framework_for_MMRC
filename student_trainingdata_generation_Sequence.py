import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import tqdm
from datasets import load_dataset


# your teacher model
model_name = "./llama-2-13b-chat-hf"
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
# print(devices)
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(device)
model = model.to(device)


prompt = 'Suppose you are a high school teacher who needs to explain reading comprehension questions to students. Now there is a reading comprehension question, and the correct answer is given you. Please analyze the reasons for each of the four options A, B, C, and D in turn, giving the reasons why the correct option is correct and the reasons why the incorrect option is incorrect.\n' \
         'Here are some examples. Please strictly follow the format of the examples to generate the results, and do not generate extra content like "Please provide the context and the question you would like me to analyze. I\'ll be happy to help you explain the correct answer and the reasons why students might have chosen other options.".\n' \
         'context: In a small village in England about 150 years ago, a mail coach was standing on the street. It didn’t come to that village often.People had to pay a lot to get a letter. The person who sent the letter didn’t have to pay the postage, while the receiver had to.“Here’s a letter for Miss Alice Brown,” said the mailman.“ I’m Alice Brown,” a girl of about 18 said in a low voice.Alice looked at the envelope for a minute, and then handed it back to the mailman.“I’m sorry I can’t take it, I don’t have enough money to pay it”, she said.A gentleman standing around were very sorry for her. Then he came up and paid the postage for her.When the gentleman gave the letter to her, she said with a smile, “ Thank you very much, This letter is from Tom. I’m goingto marry him. He went to London to look for work. I’ve waited a long time for this letter, but now I don’t need it, there isnothing in it.”“Really? How do you know that?” the gentleman said in surprise.“He told me that he would put some signs on the envelope. Look, sir, this cross in the corner means that he is well and thiscircle means he has found work. That’s good news.”The gentleman was Sir Rowland Hill. He didn’t forgot Alice and her letter.“The postage to be paid by the receiver has to be changed,” he said to himself and had a good plan.“The postage has to be much lower, what about a penny? And the person who sends the letter pays the postage. He has to buya stamp and put it on the envelope.” he said . The government accepted his plan. Then the first stamp was put out in 1840. Itwas called the “Penny Black”. It had a picture of the Queen on it.\n' \
         'question: The first postage stamp was made _.\n' \
         "options: A. in England, B. in America, C. by Alice, D. in 1910\n" \
         'answer: A\n' \
         'teacher feedback:\n' \
         'A. in England: This is the correct answer. The passage states that the first postage stamp was introduced in 1840 in England.\n'  \
         'B. in America: This option is incorrect because the passage does not mention anything about America. The events described in the passage take place in a small village in England.\n' \
         'C. by Alice: This option is incorrect because Alice is not the one who created the postage stamp. She is just a recipient of a letter who is unable to pay the postage and is helped by a gentleman, Sir Rowland Hill.\n' \
         'D. in 1910: This option is incorrect because the passage states that the first postage stamp was introduced in 1840, which is not in 1910.\n' \
         'context: Every day when I enter the classroom, I will take a look at the wall beside my seat. You will find nothing special about this old wall if you just look at it. But for the students in my class, it is a special wall. Take a good look at it, and you will get to know the real feelings and thoughts of us, the 9th graders.\nIn the middle of the wall, there is a big \"VICTORY\". It was written in pencil. I guess it must have been written by someone who got a good mark in an exam.\nA little higher above the formulas, there is a poem. It only has two sentences. It reads: All those sweet memories have disappeared. Like tears dropping in the heavy rain.           Oh! It must have been written at the end of the last semester in middle school. Classmates had to leave school and good friends had to _ . What a sad poem!\nIf you \"explore\" the wall more carefully, you will find many other interesting things, like a crying face, or a happy face, and other patterns . There are still some patterns and letters that I can\'t understand, but they all show the feeling of the students who drew them.\nFor years, the wall has witnessed  all the things that have happened in the classroom. I don\'t know how it will be next year, two years from now, or even ten years from now. But I hope more smiling faces will be drawn on it.\n' \
         'question: When has the poem been written?\n' \
         'options: A. At the beginning of the last semester B. At the end of the year C. At the end of the last semester D. At the beginning of the last month.\n' \
         'answer: C\n' \
         'teacher feedback:\n' \
         'A. At the beginning of the last semester: This option might have been chosen by students who assumed that the poem was written at the start of the semester, perhaps because the text mentions that the writer looks at the wall every day when they enter the classroom. However, the text explicitly states that the poem is "at the end of the last semester," so this option is not the correct answer.\n' \
         'B. At the end of the year: This option might have been chosen by students who assumed that the poem was written at the end of the academic year, perhaps because the text mentions that the writer is reflecting on the past semester. However, the text does not specify that the poem was written at the end of the year, so this option is not the correct answer.\n' \
         'C. At the end of the last semester: This is the correct answer! The text explicitly states that the poem was written "at the end of the last semester," so this option is the best choice.\n' \
         'D. At the beginning of the last month: This option is unlikely to have been chosen by students, as the text does not mention the beginning of the last month or any other specific time period.\n'


def generate_rationale(test_samples, model):
    inputs = tokenizer(
        test_samples,
        padding="max_length",
        truncation=True,
        # max_length=max_input_length,
        max_length=1024*2,
        return_tensors="pt",
    )

    # model = model.to(device)
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.02, pad_token_id=tokenizer.eos_token_id)

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


dataset = load_dataset('csv', data_files='./MCTEST/test.csv')
context = dataset['train']['context']
question = dataset['train']['question']
options = dataset['train']['options']
answer = dataset['train']['answer']
# rationale = dataset['train']['rationale']

with open('./MCTEST_PLUS_ORDER/test.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['context', 'question', 'options', 'answer', 'rationale'])

    num = len(question)
    # num = 5
    for n in tqdm.tqdm(range(num)):
        content = prompt + f'context: {context[n]}\n' \
                           f'question: {question[n]}\n' \
                           f'options: {options[n]}\n' \
                           f'answer: {answer[n]}\n' \
                           f'teacher feedback:'
        _, rationale = generate_rationale(content, model)
        leng = len(content)
        # print(rationale[0][leng + 1:])
        writer.writerow([context[n], question[n], options[n], answer[n], rationale[0][leng + 1:]])
