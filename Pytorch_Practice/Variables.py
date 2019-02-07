import torch
from torch.autograd import Variable # data form of weight, bias .....
import numpy as np

tensor = torch.FloatTensor([[1, 2],[3, 4]]) # tensor type
variable = Variable(tensor, requires_grad = True) # variable type

# you can do backpropagation with variable
# you can NOT do backpropagation with tensor

# input is tensor
# weight and bias are variables

print(tensor)
print(variable)

# you can use pytorch functions with both variable and tensor
t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable) # sum(x^2)/4

print(t_out)
print(v_out)

v_out.backward() # 對 x 微分
print(variable)
print(variable.grad) # d( sum(x^2)/4 )/dx = 0.5x

# ***** .data represent variable in tensor type  *******
print(variable.data.numpy()) # variable to tensor to numpy array



