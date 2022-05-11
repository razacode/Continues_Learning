from pickletools import optimize
from pyexpat import model
from tkinter import N
import torch
import torch.nn as nn

''' 3 DIFFERENT METHODS TO REMEMBER:
    torch.save(arg, PATH) # can be a model, tensor or dict
    torch.load(PATH)
    torch.load_state_dict(PATH)
'''

''' 2 DIFFERENT WAYS OF SAVING
    1) lazy way: save the whole model
    torch.save(model, PATH)

    # model class must be define somewhere
    model = torch.load(PATH)
    model.eval()

    2) recommended way: saveonly the stata_dict
    torch.save(model.state_dict(), PATH)

    # model must be created again with parameters
    model = Model(args, **kwargs)
    model.Model_state_dict(torch.load(PATH))
    model.eval()
'''

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
# this line you have to uncomment when saving model and comment when loading model
model = Model(n_input_features=6)
# trainig model ....

############### Save all #########################
# # Prinint model parameters
# # for param in model.parameters():
# #     print('Before saving parameters: ', param)

# FILE = "model_save.pth"
# # torch.save(model,FILE)

# loaded_model = torch.load(FILE)
# loaded_model.eval()

# for param in loaded_model.parameters():
#     print('After saving parameters: ', param)


############### only save_dict #########################

# FILE = "model_state_dict.pth"
# # torch.save(model.state_dict(), FILE)

# # print('printing state_dict_save model: ',model.state_dict())

# # loading state_dict saved model
# loaded_model_state = Model(n_input_features=6)
# loaded_model_state.load_state_dict(torch.load(FILE))
# loaded_model_state.eval() 
# print('printing state_dict_save model: ',loaded_model_state.state_dict())

############### load checkpoint ######################

lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

checkpoint = {
    'epoch': 90,
    'model_state': model.state_dict(),
    'optim_state': optimizer.state_dict()
}
print(optimizer.state_dict())
FILE = 'checkpoint.pth'
# torch.save(checkpoint, FILE)

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])
epoch = checkpoint['epoch']

model.eval()
print(optimizer.state_dict())


# Remember that you must call model.eval() to set dropout and batch normalization layers 
# to evaluation mode before running inference. Failing to do this will yield 
# inconsistent inference results. If you wish to resuming training, 
# call model.train() to ensure these layers are in training mode.

""" SAVING ON GPU/CPU 
# 1) Save on GPU, Load on CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)
device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
# 2) Save on GPU, Load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Note: Be sure to use the .to(torch.device('cuda')) function 
# on all model inputs, too!
# 3) Save on CPU, Load on GPU
torch.save(model.state_dict(), PATH)
device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# This loads the model to a given GPU device. 
# Next, be sure to call model.to(torch.device('cuda')) to convert the model’s parameter tensors to CUDA tensors
"""

