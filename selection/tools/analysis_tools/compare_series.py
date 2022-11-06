import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='compare two series')
    parser.add_argument('--series_a', type=str, default=None, help='series a')
    parser.add_argument('--series_b', type=str, default=None, help='series b')
    args = parser.parse_args()
    return args


def jaccard_similarity(series_a, series_b):
    """Calculate the jaccard similarity between two series."""
    intersection = len(set(series_a) & set(series_b))
    union = len(set(series_a) | set(series_b))
    return intersection / union


def main():
    args = parse_args()
    series_a = np.load(args.series_a)
    print(series_a[:50])
    series_b = np.load(args.series_b)
    print(series_b[:50])

    similarity = jaccard_similarity(series_a, series_b)
    print('Total similarity: {:.3f}'.format(similarity))

    ratios = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    for ratio in ratios:
        print('Ratio:', ratio)
        print('Quantity', int(len(series_a) * ratio))
        print('Similarity: {:.3f}'.format(
            jaccard_similarity(series_a[:int(len(series_a) * ratio)],
                               series_b[:int(len(series_b) * ratio)])))
        print('========================================================')


if __name__ == '__main__':
    main()
