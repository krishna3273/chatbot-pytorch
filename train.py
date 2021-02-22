import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from utils import tokenise,stem,bag_of_words
from model import NeuralNet



with open('intents.json','r') as f:
    intents=json.load(f)

# print(intents)

all_words=[]
tags=[]
data=[]

for intent in intents['intents']:
    # print(intent)
    tag=intent['tag']
    tags.append(tag)
    for p in intent['patterns']:
        w=tokenise(p)
        all_words.extend(w)
        data.append((w,tag))

# print(all_words)
ignore_words=['?','.',',','!']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))
# print(tags)
# print(all_words)

X_train=[]
y_train=[]
for (sentence,tag) in data:
    x=bag_of_words(sentence,all_words)
    X_train.append(x)
    label=tags.index(tag)
    y_train.append(label)

X_train=np.array(X_train)
y_train=np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.num_samples=len(X_train)
        self.x_data=X_train
        self.y_data=y_train

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.num_samples


batch_size=8
learning_rate=0.001
num_epochs=1000

dataset=ChatDataset()
loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2)

input_size=X_train.shape[1]
output_size=len(tags)
hidden_size=8
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# print(input_size,output_size,hidden_size)

model=NeuralNet(input_size,hidden_size,output_size).to(device)

loss_criterion=nn.CrossEntropyLoss()
optimiser=torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for(words,labels) in loader:
        words=words.to(device)
        labels=labels.to(device)

        output=model(words)
        loss=loss_criterion(output,labels)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    if epoch%50==0:
        print(f'epoch {epoch}/{num_epochs},loss={loss.item():.4f}')


print(f'Final loss after training={loss.item()}')

model_data={
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "all_words":all_words,
    "tags":tags
}

FILE_NAME="data.pth"
torch.save(model_data,FILE_NAME)

print(f"Model trained and parameters saved in {FILE_NAME}")