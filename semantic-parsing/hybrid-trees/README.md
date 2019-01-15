## A Generative Model for Parsing Natural Language to Meaning Representations

**Given:** A list of pairs of _natural language_ (NL) sentences and _meaning representation_ (MR) trees, which themselves consist of trees of MR productions (a CFG of sorts).

* Example MR Production: `QUERY: answer(NUM)`. Here `QUERY` is the _semantic category_, `answer` is the _function symbol_, and `NUM` is an _argument_.

**Output**: A generative model for translating NL sentences to MR representations, combined into a _hybrid tree_ structure.

The generative model is based on the following probabilities:

* **MR Model Parameters** p(m' | m, arg=k)
    * m, m' are *MR productions*.
    * This is the probability of generating the production "m'" as argument position "k" to the production "m".
* **Emission Parameters** θ(t | m, Λ)
    * m is an *MR production*, t is an *NL word/semantic category*, and Λ is the *context*.
    * This is the probability of "associating" the NL word/semantic category "t" with the production "m", with underlying context Λ.
    * The context Λ is *model specific*. In a unigram model, Λ is empty. In a bigram model, Λ is the *last word* in the hybrid sequence, preceding "t".
* **Pattern Parameters** φ(r | m)
    * m is an *MR production*, r is a *hybrid pattern* (m -> [w]Y[w], etc.)
    * This is the probability that an MR production "m" will be "expanded" into a hybrid sequence of the form "r".

We can estimate these quantities from training data using an "inside-outside" algorithm adaptation.