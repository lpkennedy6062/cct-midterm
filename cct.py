import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def load_data(path="plant_knowledge.csv"):
    df = pd.read_csv(path)
    data = df.iloc[:, 1:].values.astype(int)
    return data

def build_model(X, draws = 2000, chains = 4, tune = 1000, random_seed = 42):
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
    summary = az.summary(test, var_names = ["D", "Z"], round_to = 2)
    print("Convergence diagnostics and posterior summaries:")
    print(summary)

    D_mean = test.posterior["D"].mean(dim=("chain", "draw")).values
    print("\nPosterior mean competence for D per informant:")
    for i, d in enumerate(D_mean):
        print(f"Informant {i+1}: {d:.2}")

    Z_mean = test.posterior["Z"].mean(dim=("chain", "draw")).values
    print("\nPosterior probability consensus answer (Z=1) per question:")
    for i, z in enumerate(Z_mean):
        print(f"Q{i+1}: {z:.2f} -> consensus = {int(z>0.5)}")

    majority = X.mean(axis=0) > 0.5
    print("\nMajority vote answers per question:")
    for i, maj in enumerate(majority):
        print(f"Q{i+1}: {int(maj)} (vs CCT {int(Z_mean[i]>0.5)})")

    az.plot_posterior(test, var_names=["D"], hdi_prob = 0.95)
    plt.suptitle("Posterior Distributions of Competence (D)")
    plt.show()

    az.plot_posterior(test, var_names=["Z"], hdi_prob = 0.95)
    plt.suptitle("Posterior Distributions of Competence (Z)")
    plt.show()

def main():
    X = load_data()

    trace = build_model(X)

    analyze_results(trace, X)



if __name__ == "__main__":
    main()