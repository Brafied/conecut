import pickle
import numpy as np

with open("data/Skywork_Skywork-Reward-V2-Llama-3.1-8B_safety/nnls_0.98_COMPLETE/redundant_examples.pkl", "rb") as file:
    nnls_redundant_examples = pickle.load(file)
with open("data/Skywork_Skywork-Reward-V2-Llama-3.1-8B_safety/nnls_0.98_COMPLETE/non_redundant_examples.pkl", "rb") as file:
    nnls_non_redundant_examples = pickle.load(file)
nnls_examples = nnls_redundant_examples + nnls_non_redundant_examples
nnls_examples.sort(key=lambda x: x['index'])
with open("data/Skywork_Skywork-Reward-V2-Llama-3.1-8B_safety/nnomp_0.98_32_COMPLETE/redundant_examples.pkl", "rb") as file:
    nnomp_redundant_examples = pickle.load(file)
with open("data/Skywork_Skywork-Reward-V2-Llama-3.1-8B_safety/nnomp_0.98_32_COMPLETE/non_redundant_examples.pkl", "rb") as file:
    nnomp_non_redundant_examples = pickle.load(file)
nnomp_examples = nnomp_redundant_examples + nnomp_non_redundant_examples
nnomp_examples.sort(key=lambda x: x['index'])

# For each example, compare the reconstruction r^2 of nnls to the reconstruction r^2 of nnomp with different maximum numbers of nonzero coefficients.
differences = np.full((len(nnls_examples), 32), np.nan)
for i, (nnls_example, nnomp_example) in enumerate(zip(nnls_examples, nnomp_examples)):
    difference = nnls_example['r2_list'][0] - np.array(nnomp_example['r2_list'])
    differences[i, :len(difference)] = difference

min = np.nanmin(differences, axis=0)
q1 = np.nanpercentile(differences, 25, axis=0)
median = np.nanmedian(differences, axis=0)
q3 = np.nanpercentile(differences, 75, axis=0)
max = np.nanmax(differences, axis=0)

print('min: ', min)
print('q1: ', q1)
print('median: ', median)
print('q3: ', q3)
print('max: ', max)