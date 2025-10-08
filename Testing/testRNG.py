import numpy as np
from scipy.stats import multivariate_normal
from copy import deepcopy


np.random.seed(10)

seeds = np.random.randint(0,1000000,5)
print("Seeds:", seeds)

rng = np.random.default_rng(seeds[0])


N1 = multivariate_normal(mean=[0.0,0.0],cov=3**2,seed=rng)
noise1 = N1.rvs()
print(f"noise1: {noise1}")

N2 = multivariate_normal(mean=[0.0,0.0],cov=76**2,seed=rng)
noise2 = N2.rvs()
print(f"noise2: {noise2}")

N3 = multivariate_normal(mean=[0.0,0.0],cov=3**2,seed=rng)
noise3 = N3.rvs()
print(f"noise3: {noise3}")


print()


N1_Take2 = multivariate_normal(mean=[0.0,0.0],cov=3**2,seed=seeds[0])
noise1_Take2 = N1_Take2.rvs()
print(f"noise1_Take2: {noise1_Take2}")

N2_Take2 = multivariate_normal(mean=[0.0,0.0],cov=76**2,seed=seeds[0])
noise2_Take2 = N2_Take2.rvs()
print(f"noise2_Take2: {noise2_Take2}")

N3_Take2 = multivariate_normal(mean=[0.0,0.0],cov=3**2,seed=seeds[0])
noise3_Take2 = N3_Take2.rvs()
print(f"noise3_Take2: {noise3_Take2}")


print()
print(rng.bit_generator.state)
print(rng.bit_generator._seed_seq)
print(rng.bit_generator._seed_seq.entropy)
print()


N1_Take3 = multivariate_normal(mean=[0.0,0.0],cov=3**2,seed=rng.bit_generator._seed_seq.entropy)
noise1_Take3 = N1_Take3.rvs()
print(f"noise1_Take3: {noise1_Take3}")

N2_Take3 = multivariate_normal(mean=[0.0,0.0],cov=76**2,seed=rng.bit_generator._seed_seq.entropy)
noise2_Take3 = N2_Take3.rvs()
print(f"noise2_Take3: {noise2_Take3}")

N3_Take3 = multivariate_normal(mean=[0.0,0.0],cov=3**2,seed=rng.bit_generator._seed_seq.entropy)
noise3_Take3 = N3_Take3.rvs()
print(f"noise3_Take3: {noise3_Take3}")



rng = np.random.default_rng(seeds[0])

rng1 = deepcopy(rng)
rng2 = deepcopy(rng)

print(rng1.random())
print(rng2.random())


# rng = np.random.default_rng(seeds[0])
print(rng.random())






