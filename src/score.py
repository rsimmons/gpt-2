import json
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

import model, encoder

def score_tokens(*, hparams, tokens):
    # tokens is 1d, but model expects a batch of token-lists, so make a batch of 1
    x = tf.stack([tokens])

    lm_output = model.model(hparams=hparams, X=x, past=None, reuse=tf.AUTO_REUSE)

    # lm_output['logits'] should have shape [batch_size, tokens_length, vocab_size],
    # but from the slice in sample.py, it seemed like this might not always be the case?
    assert lm_output['logits'].shape[2] == hparams.n_vocab

    # take the first tensor, since batch size is fixed at 1
    logits = lm_output['logits'][0]
    # logits has shape [tokens_length, vocab_size]

    # get actual probabilities, in same shape as logits
    probs = model.softmax(logits)

    # The probabilities are for its guesses about the next token after each position.
    # We want to look up the probability that it gave for what actually turned out to be the "true"
    # next token.
    next_tokens = tokens[1:]
    tokens_range = tf.range(tf.shape(next_tokens)[0])
    indices = tf.stack([tokens_range, next_tokens], axis=-1)
    # indices has shape [next_tokens_length, 2]. it is a list of [pos, token] that we want to lookup in probs
    probs_next = tf.gather_nd(probs, indices)
    # probs_next has shape [tokens_length-1], and has the predicted probability of each input token (after the first one)

    # Get log probabilities
    ln_probs_next = tf.log(probs_next)

    return ln_probs_next

def score_texts(*, model_name, texts, exclude_end, models_dir='models'):
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    end_token = enc.encoder['<|endoftext|>']
    start_token = end_token # it does double duty

    with tf.Session(graph=tf.Graph()) as sess:
        tokens_tensor = tf.placeholder(tf.int32, [None])

        output = score_tokens(hparams=hparams, tokens=tokens_tensor)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        for text in texts:
            # prepend the start token so that we get a probability for the first "real" token
            tokens = enc.encode(text)
            if not exclude_end:
                tokens += [end_token]
            tokens_with_start = [start_token] + tokens

            logprobs = sess.run(output, feed_dict={
                tokens_tensor: tokens_with_start,
            })

            logprobs_list = logprobs.tolist()
            assert len(logprobs_list) == len(tokens) # sanity check

            print('%s\t%.5g' % (text, sum(logprobs_list)))
            for t, lp in zip(tokens, logprobs_list):
                print('%s\t%.5g' % (enc.decoder[t], lp))
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('--model', default='124M')
    parser.add_argument('--exclude-end', action='store_true')
    args = parser.parse_args()

    if args.input_file == '-':
        input_f = sys.stdin
    else:
        input_f = open(args.input_file, 'r')

    texts = []
    for line in input_f:
        sline = line.strip()
        if not sline:
            continue
        texts.append(sline)

    score_texts(model_name=args.model, texts=texts, exclude_end=args.exclude_end)
