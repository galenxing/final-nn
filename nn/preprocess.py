# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    n_examples_per_class = int(np.mean(np.unique(labels, return_counts=True)[1]))

    samples_data = []
    samples_label = []

    seqs = np.array(seqs)
    labels = np.array(labels)
    for label in np.unique(labels):

        tmp_labels = labels[labels == label]
        tmp_seqs = seqs[labels == label]

        selected = np.random.choice(len(tmp_labels), n_examples_per_class)
        samples_data.append(tmp_seqs[selected])
        samples_label.append(tmp_labels[selected])

    samples_data = np.concatenate(samples_data)
    samples_label = np.concatenate(samples_label)
    return samples_data, samples_label
    

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    for i,s in enumerate(seq_arr):
        s = s.replace('A', '1000')
        s = s.replace('T', '0100')
        s = s.replace('C', '0010')
        s = s.replace('G', '0001')
        s = list(s)
        seq_arr[i] = s

    return seq_arr
    
    