import os
import ast
from collections import Counter
from scipy import sparse
import random

import vocabulary_with_counts
import file_handling as fh

class FeatureExtractorCounts:

    dirname = None
    name = None
    prefix = None
    n = None
    min_doc_threshold = None
    binarize = None

    index = None
    vocab = None

    def __init__(self, basedir, name, prefix, min_doc_threshold=1, binarize=True):
        self.name = name
        self.prefix = prefix
        self.min_doc_threshold = int(min_doc_threshold)
        self.binarize = ast.literal_eval(str(binarize))
        self.feature_counts = None
        self.index = None
        self.vocab = None
        self.make_dirname(basedir)

    def get_name(self):
        return self.name

    def get_prefix(self):
        return self.prefix

    def get_min_doc_threshold(self):
        return self.min_doc_threshold

    def get_binarize(self):
        return self.binarize

    def get_dirname(self):
        return self.dirname

    def make_dirname(self, basedir):
        dirname = ','.join([self.name, 'mdt=' + str(self.min_doc_threshold), 'bin=' + str(self.binarize)])
        self.dirname = os.path.join(basedir, dirname)

    def get_feature_filename(self):
        return fh.make_filename(fh.makedirs(self.dirname), 'counts', 'pkl')

    def get_oov_count_filename(self):
        return fh.make_filename(fh.makedirs(self.dirname), 'oov_counts', 'json')

    def make_vocabulary(self, tokens, items, verbose=True):
        if verbose:
            print "Making vocabulary for", self.get_name()

        vocab = vocabulary_with_counts.VocabWithCounts(self.get_prefix(), add_oov=True)

        for item in items:
            vocab.add_tokens(tokens[item])

        if verbose:
            print "Vocabulary size before pruning:", len(vocab)

        vocab.prune(min_doc_threshold=self.get_min_doc_threshold())

        if verbose:
            print "Vocabulary size after pruning:", len(vocab)

        return vocab

    def extract_feature_counts(self, items, tokens, vocab):
        n_items = len(items)
        n_features = len(vocab)

        row_starts_and_ends = [0]
        column_indices = []
        values = []
        oov_counts = []

        for item in items:
            #token_counts = Counter(tokens[item])
            #token_keys = token_counts.keys()
            #token_indices = vocab.get_indices(token_keys)

            # get the index for each token
            token_indices = vocab.get_indices(tokens[item])

            # count how many times each index appears
            token_counter = Counter(token_indices)
            token_keys = token_counter.keys()
            token_counts = token_counter.values()

            # put it into the from of a sparse matix
            column_indices.extend(token_keys)
            if self.get_binarize():
                values.extend([1]*len(token_counts))
            else:
                values.extend(token_counts)

            oov_counts.append(token_counter.get(vocab.oov_index, 0))
            row_starts_and_ends.append(len(column_indices))

        dtype = 'int32'

        feature_counts = sparse.csr_matrix((values, column_indices, row_starts_and_ends), dtype=dtype)

        assert feature_counts.shape[0] == n_items
        assert feature_counts.shape[1] == n_features

        return feature_counts, oov_counts

    def load_from_files(self, debug=False, debug_index=None):
        vocab = vocabulary_with_counts.VocabWithCounts(self.get_prefix(), add_oov=True,
                                                       read_from_filename=self.get_vocab_filename())
        index = fh.read_json(self.get_index_filename())
        feature_counts = fh.unpickle_data(self.get_feature_filename())
        oov_counts = fh.read_json(self.get_oov_count_filename())

        # TESTING
        if debug:
            if debug_index is None:
                item_index = random.randint(0, len(index))
            else:
                item_index = debug_index
            item = index[item_index]
            counts = feature_counts[item_index, :]

            print item
            print counts.indices
            print counts.data
            print vocab.get_tokens(counts.indices)
            print oov_counts[item_index]

        self.feature_counts = feature_counts
        self.index = index
        self.vocab = vocab
        self.oov_counts = oov_counts

    def get_counts(self):
        counts = self.feature_counts
        column_names = self.vocab.index2token
        return self.index, column_names, counts

    def get_vocab_filename(self):
        return fh.make_filename(fh.makedirs(self.get_dirname()), 'vocab', 'json')

    def get_index_filename(self):
        return fh.make_filename(fh.makedirs(self.get_dirname()), 'index', 'json')



