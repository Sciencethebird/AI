import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data) # numpy array to torch data
tensor2array = torch_data.numpy() # torch data to numpy


print(
    '\nNumpy', np_data,
    '\ntorch', torch_data,
    '\nNumpy', tensor2array
)

data = [[1, 2],[3, 4]]
tensor = torch.FloatTensor(data)
nparray = np.array(data)

print(
    '\nNumpy: ', nparray.dot(nparray), # matrix multiplication
    '\nNumpy: ', np.matmul(nparray, nparray), # matrix multiplication
    '\ntorch: ', tensor.mm(tensor),
    #'\ntorch(dot): ', tensor.dot(3)
)