import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline
import json

def fine_tune_model(base_model, custom_dataset):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    instructions = [entry["instruction"] for entry in custom_dataset]
    outputs = [entry["output"] for entry in custom_dataset]

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    inputs = tokenizer(instructions, outputs,  return_tensors='pt', truncation=True, padding=True)


    training_args = TrainingArguments(output_dir='./model', num_train_epochs=3, per_device_train_batch_size=16, 
    per_device_eval_batch_size=64, eval_steps=400, save_steps=800, warmup_steps=500)
    # it will save the fine tuned model to a model folder which i use in the chat_with_bot function later on.
    trainer = Trainer(model=model, args=training_args, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), train_dataset=inputs['input_ids'])
    trainer.train()
    

def chat_with_bot(model_path):
    chat_pipeline = pipeline('text-generation', model=model_path)
    st.title("getfit chatbot")
    if "messages" not in st.session_state:
      st.session_state.messages = []


    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])


    if prompt := st.chat_input("Hey, i am a fitness assistant, ask me anything"):
    
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        with st.chat_message("user"):
          st.markdown(prompt)
        with st.chat_message("assistant"):
          response = chat_pipeline(prompt)[0]['generated_text']
          st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})    
custom_dataset_path = "./dataset/fitness-chat-prompt-completion-dataset.json"
# this dataset is taken and downloaded from huggingface datasets. https://huggingface.co/datasets/chibbss/fitness-chat-prompt-completion-dataset/tree/main
with open(custom_dataset_path, "r", encoding="utf-8") as file:
    custom_dataset = json.load(file)

base_model = 'gpt2'
#using gp2 base model
fine_tune_model(base_model, custom_dataset)

model_path = './model'

chat_with_bot(model_path)
