import tensorflow as tf
import numpy as np
import re
import sys
import os
from shutil import copyfile
import fire

SIZE_PER_HEAD = int(768 / 12) # This is only true for BERT-Base

def params_for_attn(ledger, layer, masks=False):
    end = 'mask' if masks else 'weights'
    return (
        ledger[f'bert/encoder/layer_{layer}/attention/self/key/{end}'],
        ledger[f'bert/encoder/layer_{layer}/attention/self/query/{end}'],
        ledger[f'bert/encoder/layer_{layer}/attention/self/value/{end}'],
        ledger[f'bert/encoder/layer_{layer}/attention/output/fully_connected/{end}'],
        )

def extract_single_head(key, query, value, FC, head_ind):
    assert all([tensor.shape == (768, 768) for tensor in [key, query, value, FC]])

    return tuple(tensor[:,SIZE_PER_HEAD*head_ind:SIZE_PER_HEAD*(head_ind+1)]
                 for tensor in [key, query, value]
                 ) + (FC[SIZE_PER_HEAD*head_ind:SIZE_PER_HEAD*(head_ind+1),:],)

def prune_attn_head_uniform(key, query, value, FC, key_mask, query_mask, value_mask, FC_mask, sparsity):
    key_mask[:, :] = 1
    query_mask[:, :] = 1
    value_mask[:, :] = 1
    FC_mask[:, :] = 1

    sample_size = int(SIZE_PER_HEAD * (1 - sparsity))
    indices = np.random.choice(SIZE_PER_HEAD, size=sample_size, replace=False)
    key_mask[:, indices] = 0
    query_mask[:, indices] = 0
    value_mask[:, indices] = 0
    FC_mask[indices, :] = 0


def prune_multihead_attn(model_dir, out_dir, method, sparsity: float):
    """Prunes [sparsity] of the number of attention heads in [model_dir].
    Makes a new checkpoint [out_dir]
    """
    model_dir = model_dir.rstrip('/')

    with tf.Session() as sess:

        # Load all the variables from the checkpoint
        ledger = {}
        print("Loading tensors...")
        for var_name, _ in tf.train.list_variables(model_dir):
            ledger[var_name] = tf.contrib.framework.load_variable(model_dir, var_name)

        print("Pruning...")
        for layer in range(12):
            params = params_for_attn(ledger, layer)
            masks = params_for_attn(ledger, layer, masks=True)
            for head in range(12):
                head_params = extract_single_head(*params, head)
                head_masks = extract_single_head(*masks, head)

                if method == "uniform":
                    prune_attn_head_uniform(*head_params, *head_masks, sparsity)
                else:
                    raise NotImplementedError()

        print("Saving Checkpoint...")
        for var_name, var_tensor_np in ledger.items():
            var = tf.Variable(var_tensor_np, name=var_name)

        # Save these new variables
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        os.mkdir(out_dir)
        saver.save(sess, os.path.join(out_dir, 'pruned.ckpt'))

if __name__ == '__main__':
    fire.Fire(prune_multihead_attn)
