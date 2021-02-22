import random
import json
import torch
from model import NeuralNet
from utils import bag_of_words,tokenise

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    intents=json.load(f)

model_data=torch.load("data.pth")

input_size=model_data["input_size"]
hidden_size=model_data["hidden_size"]
output_size=model_data["output_size"]
all_words=model_data["all_words"]
tags=model_data["tags"]
model_state=model_data["model_state"]

model=NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

name="kmt_bot"
print("Let's chat! ('quit' to exit)" )
tag=""
while True:
    sentence=input('You: ')
    if sentence=="quit":
        break
    sentence=tokenise(sentence)
    vec=bag_of_words(sentence,all_words)
    vec=torch.from_numpy(vec.reshape(1,vec.shape[0]))

    output=model(vec)
    _,predicted=torch.max(output,dim=1)
    tag=tags[predicted.item()]

    prob=torch.softmax(output,dim=1)[0][predicted.item()]
    if prob.item()>0.75:
        for intent in intents['intents']:                
            if tag==intent["tag"]:
                random_res=random.choice(intent["responses"])
                print(f"{name}: {random_res}")
    else:
        print(f"{name}: Sorry,I cannot understand your question")

    if tag=="goodbye":
        break