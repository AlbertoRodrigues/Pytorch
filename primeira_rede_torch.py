import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import torch.optim as optim

dados=pd.read_csv("auto_mpg_ajeitado.csv")
x=dados.drop("mpg",axis=1)
y=dados["mpg"]

from sklearn.model_selection import train_test_split
x_treino,x_teste,y_treino,y_teste=train_test_split(x,y,test_size=0.25)
x_treino=torch.from_numpy(x_treino.values).float()
x_teste=torch.from_numpy(x_teste.values).float()
y_treino=torch.from_numpy(y_treino.values).float()
y_teste=torch.from_numpy(y_teste.values).float()


class modelo(nn.Module):
    def __init__(self):
        super(modelo, self).__init__()
    
        self.entrada=nn.Linear(7,100)
        self.camada1=nn.Linear(100,80)
        self.saida=nn.Linear(80,1)
    
    def forward(self,X):
        X=nn.BatchNorm1d(7)(X)
        X=nn.ReLU()(self.entrada(X))
        X=nn.ReLU()(self.camada1(X))
        y=self.saida(X)
        
        return y

rede_neural=modelo()
    
erro=nn.MSELoss()
otimizador=optim.SGD(rede_neural.parameters(), lr=0.01)

for i in range(100):
    otimizador.zero_grad()
    predicao=rede_neural(x_treino)
    perda=erro(predicao,y_treino)
    perda.backward()
    otimizador.step()
    
    print("Iteração ",i+1,"Perda: ",torch.sqrt(perda))

#Predicao
rede_neural(x_teste)    
torch.sqrt(erro(rede_neural(x_teste),y_teste))