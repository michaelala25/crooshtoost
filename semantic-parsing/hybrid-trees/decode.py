# THIS IS JUST PSEUDOCODE.
# This also assumes the generative model parameters have already been calculated.

def decode_1(nl_sent):
    # Naive approach (which doesn't make sense).

    # The reason this doesn't make sense is because we would never iterate over all possible MR representations.
    # This is akin to iterating over all state sequences in an HMM to perform Viterbi decoding, which is dumb.
    return max(
        mr_representations(nl_sent), 
        key=lambda m: max(
            map(lambda T: P(w, m, T), hybrid_trees_with_yield(m, w))
            )
        )

def decode_2(nl_sent):
    # The paper doesn't explicitly state how to decode an nl sentence, so we have to innovate
    pass