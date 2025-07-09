pip install --upgrade datasets -q
pip uninstall fsspec -y -q
pip install fsspec -q        #I was geeting error bcz of fsspec aur datasets version like the load_dataset was throwing an error 
pip install datasets -q

from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print(dataset["train"][1])       #row with index 0 was empty which returned{'text':''}

pip install transformers datasets accelerate -q   #without accelerate trainer api was throwing error

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch      

from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

train_dataset = dataset["train"].select(range(30000))      #the dataset was so big that it was taking a lot of time on cpu so i reduced the sample size of both train and validation class
val_dataset = dataset["validation"].select(range(1000))      #to run it fast i used colab gpu set up

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token       #gpt2 doesnt have built in padding tokens

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["text"])    #apply tokenize function on the dataset and remove raw text keeps only tokenized input

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    eval_strategy="no",
    save_strategy="no",
    logging_steps=10,
    report_to="none"
)        #eval_strategy could not be recognized by the transformer and throws error due to which i upgraded my transformer version

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()     #was taking so much time on jupyter notebook(bcz of no gpu) so i shifted to colab to do it fast

import math

def compute_perplexity(eval_loss):       #function to compute perplexity
    return math.exp(eval_loss)

eval_results = trainer.evaluate()
print("Eval loss:", eval_results["eval_loss"])
print("Perplexity:", compute_perplexity(eval_results["eval_loss"]))

input_text = "Area of a circle is"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}      #its to move the inputs to where the model is present(cpu or gpu)
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True)      #do_sample adds randomness to the prediction so that the output would be different in each run
print(tokenizer.decode(outputs[0]))