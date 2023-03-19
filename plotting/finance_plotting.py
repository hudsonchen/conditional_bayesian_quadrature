import numpy as np
import matplotlib.pyplot as plt
import pickle

file = open('./results/finance/BMC_mean', 'rb')
BMC_mean = pickle.load(file)
file = open('./results/finance/BMC_std', 'rb')
BMC_std = pickle.load(file)
file = open('./results/finance/poly', 'rb')
poly = pickle.load(file)
file = open('./results/finance/importance_sampling', 'rb')
importance_sampling = pickle.load(file)
file = open('./results/finance/mean_shrinkage', 'rb')
mean_shrinkage = pickle.load(file)

true_value = 9.02
Nx_all = BMC_mean.keys()

pause = True