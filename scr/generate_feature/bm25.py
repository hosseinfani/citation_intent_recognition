from __future__ import absolute_import, division, print_function, unicode_literals
import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('bm25')

import gensim as gs
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
import re
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import _document_frequency

class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
        return self

    def transform(self, X, copy=True):
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            X = sp.csr_matrix(X, copy=copy)
        else:
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        dl = X.sum(axis=1)
        sz = X.indptr[1:] - X.indptr[0:-1]
        rep = np.repeat(np.asarray(dl), sz)
        avgdl = np.average(dl)
        data = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag
        return X

def bm25(train_file, validation_file, papers_file, ngrams, output_folder):
    logger.info('loading training data ...')
    train = pd.read_csv(train_file, names=['description_id', 'paper_id', 'description_text'],encoding='ISO-8859-1')#, header=0)#.sample(n=100)
    n_train = len(train)
    logger.info('{} training data has been loaded.'.format(n_train))

    logger.info('loading validation data ...')
    valid = pd.read_csv(validation_file, names=['description_id', 'description_text'],encoding='ISO-8859-1')#, header=0)#.sample(n=100)
    n_valid = len(valid)
    logger.info('{} validation data has been loaded.'.format(n_valid))

    logger.info('loading papers ...')
    papers = pd.read_csv(papers_file, names=['abstract', 'journal', 'keywords', 'paper_id', 'title', 'year'], encoding='ISO-8859-1', na_values=['NO_CONTENT', 'no-content'])#, header=0)#.sample(n=100)
    n_papers = len(papers)
    logger.info('{} papers have been loaded.'.format(n_papers))

    preprocessing_funcs = [gs.parsing.strip_tags,
                           gs.parsing.strip_punctuation,
                           gs.parsing.strip_multiple_whitespaces,
                           gs.parsing.strip_numeric,
                           gs.parsing.remove_stopwords,
                           gs.parsing.strip_short,
                           gs.parsing.strip_multiple_whitespaces,
                           gs.parsing.stem_text]

    logger.info('preprocessig on texts ...')
    train['processed_description'] = train.apply(lambda r: ' '.join(gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs, s=r['description_text'].lower())),deacc=True, min_len=3, max_len=15)), axis=1)
    valid['processed_description'] = valid.apply(lambda r: ' '.join(gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs, s=r['description_text'].lower())),deacc=True, min_len=3, max_len=15)), axis=1)
    papers['processed_abstract'] = papers.apply(lambda r: ' '.join(gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs, s=r['abstract'].lower())), deacc=True,min_len=3, max_len=15) if not pd.isnull(r['abstract']) else ['NaN']), axis=1)
    papers['processed_journal'] = papers.apply(lambda r: ' '.join(gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs, s=r['journal'].lower())), deacc=True,min_len=3, max_len=15) if not pd.isnull(r['journal']) else ['NaN']), axis=1)
    papers['processed_title'] = papers.apply(lambda r: ' '.join(gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs, s=r['title'].lower())), deacc=True,min_len=3, max_len=15)), axis=1)

    texts = pd.concat([train['processed_description'],
                      valid['processed_description'],
                      papers['processed_abstract'],
                      papers['processed_journal'],
                      papers['processed_title']])

    for ngram in ngrams:
        logger.info('building {}-gram tc matrix ...'.format(ngram))
        cv = CountVectorizer(ngram_range=ngram)
        tcs = cv.fit_transform(pd.concat([train['processed_description'],
                                              valid['processed_description'],
                                              papers['processed_abstract'],
                                              papers['processed_journal'],
                                              papers['processed_title']]))

        bm = BM25Transformer(k1=2.0, b=0.75).fit(tcs).transform(tcs)
        logger.info('saving bm25 of {}-gram ...'.format(ngram))
        joblib.dump(bm[0: n_train], output_folder + 'train.bm.{}_{}_gram'.format(ngram[0], ngram[1]))
        joblib.dump(bm[n_train: n_train + n_valid], output_folder + 'valid.bm.{}_{}_gram'.format(ngram[0], ngram[1]))
        joblib.dump(bm[n_train + n_valid: n_train + n_valid + n_papers], output_folder + 'abstract.bm.{}_{}_gram'.format(ngram[0], ngram[1]))
        joblib.dump(bm[n_train + n_valid + n_papers: n_train + n_valid + 2 * n_papers], output_folder + 'journal.bm.{}_{}_gram'.format(ngram[0], ngram[1]))
        joblib.dump(bm[n_train + n_valid + 2 * n_papers:], output_folder + 'title.bm.{}_{}_gram'.format(ngram[0], ngram[1]))

def test():
    bm25(train_file="../../../dataset/hossein_sample_train.txt",
         validation_file="../../../dataset/hossein_sample_valid.txt",
         papers_file="../../../dataset/hossein_sample_papers.txt",
         ngrams=[(1, 1), (1, 2)],
         output_folder="../output/features/")


if __name__ == '__main__':
    test()
    exit(0)
    # path = 'D:/output/'
    path = '/mnt/sata_disk/hossein/wsdmcup2020/output/'
    bm25(train_file="../../../dataset/train.csv",
        validation_file="../../../dataset/validation.csv",
        papers_file="../../../dataset/papers.csv",
        ngrams=[(1,1),(1,2)],
        output_folder= path + "features/")
