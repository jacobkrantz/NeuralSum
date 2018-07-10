from __future__ import print_function

from tensor2tensor.data_generators import text_encoder

"""
Manually copy in tensor values representing a sentence or summary. Uses
    SubwordTextEncoder to tensor to a human readable version with the
    original words.
    Tensors pulled from NeuralSum/summary_modalities.py loss().
"""

def visualize_target():
    vocab_file = "../data/tensor2tensor/data/vocab.summary_problem_small.32768.subwords"
    txt_encoder = text_encoder.SubwordTextEncoder(vocab_file)
    targets = [
        175, 172, 139, 29, 44, 117, 473, 2, 168,
        6181, 15401, 7, 6, 2814, 72, 291, 2281,
        6975, 138, 22, 8, 1907, 2196, 52, 28, 1374,
        202, 991, 1271, 5722, 3, 2, 175, 5, 1, 0,
        175, 172, 3, 473, 168, 72, 291, 3095, 1
    ]
    t1 = [
        72, 172, 11, 10, 5725, 1938, 996, 140, 14,
        27, 44, 50, 279, 39, 13, 2481, 20, 4672,
        1938, 877, 4, 883, 3, 2023, 20, 3145, 810,
        916, 15, 965, 1938, 5, 1, 0, 5725, 3, 901,
        1938, 3248, 4, 883, 1, 0, 0, 0
    ]
    print(txt_encoder.decode(targets))
    print(txt_encoder.decode(t1))

if __name__ == '__main__':
    visualize_target()
