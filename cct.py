import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def load_data(path="plant_knowledge.csv"):
    '''Loads plant data in csv format.'''
    df = pd.read_csv(path)
    data = df.iloc[:, 1:].values.astype(int)
    return data

def build_model(X, draws = 2000, chains = 4, tune = 1000, random_seed = 42):
    '''Builds the CCT PyMC model, draws posterior samples, and returns the inferred data.'''
    N, M = X.shape
    with pm.Model() as model:
        D = pm.Uniform("D", lower = 0.5, upper = 1.0, shape = N)
        Z = pm.Bernoulli("Z", p = 0.5, shape=M)
        D_col = D[:, None]
        Z_row = Z[None, :]
        p = Z_row * D_col + (1 - Z_row) * (1 - D_col)
        X_obs = pm.Bernoulli("X_obs", p = p, observed = X)
        test = pm.sample(draws = draws, chains = chains, tune = tune, random_seed = random_seed, return_inferencedata=True)
    return test

def analyze_results(test, X):
    '''Prints convergence diagnostics, competence estimates, concensus answers, and compares to majority vote. 
    It will also plot the posterior distributions.'''
    #Summary with R-hat values
    summary = az.summary(test, var_names = ["D", "Z"], round_to = 2)
    print("Convergence diagnostics and posterior summaries:")
    print(summary)

    #Posterior mean competence
    D_mean = test.posterior["D"].mean(dim=("chain", "draw")).values
    print("\nPosterior mean competence for D per informant:")
    for i, d in enumerate(D_mean):
        print(f"Informant {i+1}: {d:.2}")

    #Posterior probability of consensus
    Z_mean = test.posterior["Z"].mean(dim=("chain", "draw")).values
    print("\nPosterior probability consensus answer (Z=1) per question:")
    for j, z in enumerate(Z_mean):
        print(f"Q{j+1}: {z:.2f} -> consensus = {int(z>0.5)}")

    #Compares to a simple majority vote
    majority = X.mean(axis=0) > 0.5
    print("\nMajority vote answers per question:")
    for j, maj in enumerate(majority):
        print(f"Q{j+1}: {int(maj)} (vs CCT {int(Z_mean[i]>0.5)})")

    # Plots distributions for D
    az.plot_posterior(test, var_names=["D"], hdi_prob = 0.95)
    plt.suptitle("Posterior Distributions of Competence (D)")
    plt.show()

    # Plots distributions for Z
    az.plot_posterior(test, var_names=["Z"], hdi_prob = 0.95)
    plt.suptitle("Posterior Distributions of Competence (Z)")
    plt.show()

def main():
    X = load_data()

    trace = build_model(X)

    analyze_results(trace, X)

hello = load_data()

if __name__ == "__main__":
    main()



'''Report:
I built a simple model in PyMC with each person's Di was assumed to be between 0.5 and 1 so that nobody's competence was less than a coin flip. 
I also looked at making sure that the true answer to any of the Zj questions was either 0 or 1. This mean that if the true answer was 1, an informant would answer 1 with the probability of Di.
If the right answer was 0, then they would answer as 1 - Di. I ran 4 MCMC tests with 4 chains and 2000 draws per chain. I also added tuning steps as well into my equations. 
This meant that I was able to get accurate estimates reliably without any divergences. 

When looking at my results, I had the average competence scores to be between 0.56 and 0.87, showing that some people were much more relidable than others. 
If the average was 0.5 it would simply be guessing. When looking at the consensus probabilities, it was almost an even split between the 0s and 1s with the majority of questions having a clear consensus except mainly for question 5 which was at 0.58.
Comparison with a simple majority vote more often than not matched the model with the raw majority exactly 16/20 times. 
The other 4 questions were flipped with the more competent informants answering 1 but overall 1 was answered much less.
Overall, this model is able to show us which informants are most knowledgeable and produces a consensus key that mostly argrees with the average vote. 
This demonstrates how by weighting by competence rather than treating each response equally, we can get a better sense of a more accurate group decision rather than treating each response the same.
'''