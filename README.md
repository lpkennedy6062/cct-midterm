Cultural Consensus Theory with PyMC

Liam Kennedy





Report:
I built a simple PyMC model with each person's competence (Di) was constrained between 0.5 and 1 so that nobody's competence was less than a coin flip. 
I also looked at making sure that the true answer to any of the Zj questions was either 0 or 1 Bernoulli with a 50/50 prior. This meant that if the true answer was 1, an informant would answer 1 with the probability of Di.
If the right answer was 0, then they would answer as (1 - Di). I ran 4 MCMC tests with 4 chains and 2000 draws per chain. I also added tuning steps as well into my equations. 
This meant that I was able to get accurate estimates reliably without any divergences. 

When looking at my results, I had the average competence scores to be between 0.56 and 0.87, showing that some people were much more in sync with the group than others. 
For example, if the average was 0.5 it would simply be guessing. 
When looking at the consensus probabilities, it was almost an even split between the 0s and 1s with the majority of questions having a clear consensus except mainly for question 5 which was at 0.58.
Comparison with a simple majority vote more often than not matched the model with the raw majority exactly 16/20 times. 
The other 4 questions were flipped with the more competent informants answering 1 but overall 1 was answered much less.
Overall, this model is able to show us which informants are most knowledgeable and produces a consensus key that mostly argrees with the average vote. 
This demonstrates how by weighting by competence rather than treating each response equally, we can get a better sense of a more accurate group decision rather than treating each response and responder the same.
