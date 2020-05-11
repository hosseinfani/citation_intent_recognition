import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('d2v')

import gensim as gs
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
import re
from sklearn.externals import joblib

def d2v(train_file, validation_file, papers_file, output_folder, dims):
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
    train['processed_description'] = train.apply(lambda r: gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs, s=r['description_text'].lower())),deacc=True, min_len=3, max_len=15), axis=1)
    valid['processed_description'] = valid.apply(lambda r: gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs, s=r['description_text'].lower())),deacc=True, min_len=3, max_len=15), axis=1)
    papers['processed_abstract'] = papers.apply(lambda r: gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs, s=r['abstract'].lower())), deacc=True,min_len=3, max_len=15) if not pd.isnull(r['abstract']) else ['NaN'], axis=1)
    papers['processed_journal'] = papers.apply(lambda r: gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs, s=r['journal'].lower())), deacc=True,min_len=3, max_len=15) if not pd.isnull(r['journal']) else ['NaN'], axis=1)
    papers['processed_title'] = papers.apply(lambda r: gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs, s=r['title'].lower())), deacc=True,min_len=3, max_len=15), axis=1)

    texts = pd.concat([train['processed_description'],
                      valid['processed_description'],
                      papers['processed_abstract'],
                      papers['processed_journal'],
                      papers['processed_title']])

    docs = []
    for index, text in enumerate(texts):
        docs.append(gs.models.doc2vec.TaggedDocument(text, [index]))

    for dim in dims:
        d2v = gs.models.Doc2Vec(dm=1, vector_size=dim, window=5, min_alpha=0.025, workers=multiprocessing.cpu_count())
        d2v.build_vocab(docs)
        for e in range(100):#epoch
            if not (e % 10):
                print('iteration {} of d2v of size {}'.format(e, dim))
            d2v.train(docs, total_examples=d2v.corpus_count, epochs=d2v.epochs)
            d2v.alpha -= 0.002  # decrease the learning rate
            d2v.min_alpha = d2v.alpha  # fix the learning rate, no decay

        logger.info('saving d2v of size {} ...'.format(dim))
        joblib.dump(d2v.docvecs.vectors_docs[0: n_train], output_folder + 'train.d2v.{}'.format(dim))
        joblib.dump(d2v.docvecs.vectors_docs[n_train: n_train + n_valid], output_folder + 'valid.d2v.{}'.format(dim))
        joblib.dump(d2v.docvecs.vectors_docs[n_train + n_valid: n_train + n_valid + n_papers], output_folder + 'abstract.d2v.{}'.format(dim))
        joblib.dump(d2v.docvecs.vectors_docs[n_train + n_valid + n_papers: n_train + n_valid + 2 * n_papers], output_folder + 'journal.d2v.{}'.format(dim))
        joblib.dump(d2v.docvecs.vectors_docs[n_train + n_valid + 2 * n_papers:], output_folder + 'title.d2v.{}'.format(dim))

def test():
    d2v(train_file="../../../dataset/hossein_sample_train.txt",
        validation_file="../../../dataset/hossein_sample_valid.txt",
        papers_file="../../../dataset/hossein_sample_papers.txt",
        output_folder="../output/features/",
        dims=[50, 100, 200, 300])


if __name__ == '__main__':
    test()
    exit(0)
    # path = 'D:/output/'
    path = '/mnt/sata_disk/hossein/wsdmcup2020/output/'
    d2v(train_file="../../../dataset/train.csv",
        validation_file="../../../dataset/validation.csv",
        papers_file="../../../dataset/papers.csv",
        output_folder= path + "features/",
        dims=[50, 100, 200, 300])
