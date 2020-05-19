import fire
import json
import numpy as np
import sys

def layers(first_fname, second_fname):
    """Returns an iterator over pairs of layer activations from BERT feature files"""
    first_file = open(first_fname)
    second_file = open(second_fname)
    for first_line, second_line in zip(first_file, second_file):
        first_json = json.loads(first_line)
        second_json = json.loads(second_line)

        for first_feature, second_feature in zip(first_json['features'], second_json['features']):
            assert first_feature['token'] == second_feature['token']

            for first_layer, second_layer in zip(first_feature['layers'], second_feature['layers']):
                assert first_layer['index'] == second_layer['index']

                first_vec = np.array(first_layer['values'])
                second_vec = np.array(second_layer['values'])

                yield first_vec, second_vec

def main(first_fname, second_fname):
    total_error = 0
    count = 0

    for first_vec, second_vec in layers(first_fname, second_fname):
        total_error += np.sum(np.abs(first_vec - second_vec))
        count += 1
        if count % 10000 == 0:
            print(count, file=sys.stderr)
        if count > 300000:
            print('Passed 300k layers, ignoring the rest.', file=sys.stderr)
            break

    print(f'{total_error} / {count} = {total_error / count}')


if __name__ == '__main__':
    fire.Fire(main)
