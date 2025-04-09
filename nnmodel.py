import torch.nn as nn
import torch.nn.functional as F

class DaNN(nn.Module):
    def __init__(self, n_input=3, n_hidden1=256, n_hidden2=256, n_class=88):
        super(DaNN, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden1)
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(n_hidden1, n_hidden2)
        self.layer_hidden2 = nn.Linear(n_hidden2, n_class)

    def forward(self, x, params):
        if params is not None:
            #print("TRUE")

            #print(list(self.parameters())[5])
            i = 0
            for p_self,p_meta in zip(self.parameters(), params):
                #i = i + 1
                #print(i)
                #if (i == 6):
                #    print(p_self.data)
                #    print(p_meta.data)
                p_self.data =  p_meta.data
                
        #else:
        #    print("False")
        x1 = self.layer_input(x)
        x2 = self.relu(x1)
        x3 = self.layer_hidden(x2)
        x4 = self.relu(x3)
        y1 = self.layer_hidden2(x4)
        y = F.log_softmax(y1,dim=1)
      #  y2 = F.log_softmax(y1/10)
        return y   
