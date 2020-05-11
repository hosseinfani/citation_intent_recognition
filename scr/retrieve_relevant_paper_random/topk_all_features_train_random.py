import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('top_k_random')

import pandas as pd
from sklearn.externals import joblib
from sklearn import metrics
import numpy as np

def topk(similarity_file, train, papers, output, random_sample_idx, k=10):
    logger.info('loading similarity file {} ...'.format(similarity_file))
    sims = joblib.load(similarity_file)

    logger.info('finding top-{} papers for {} ...'.format(k, similarity_file))
    ind = np.argsort(sims, axis=1)[:, :-(k+1):-1]
    scores = np.take_along_axis(sims, ind, axis=1)
    topks = pd.DataFrame()
    for i, topk in enumerate(ind[:, :k]):
        topks = topks.append({'paper_id': train.paper_id[random_sample_idx[i]], 'topk': ','.join(papers.paper_id[topk]), 'score':','.join(str(s) for s in scores[i, :k])}, ignore_index=True)

    logger.info('saving top-{} relevant papers for {} ...'.format(k, similarity_file))
    topks.to_csv(output, index=False)

def test():
    path = '../output/'
    logger.info('loading papers ...')
    papers = pd.read_csv('../../../dataset/hossein_sample_papers.txt', names=['abstract', 'journal', 'keywords', 'paper_id', 'title', 'year'], encoding='ISO-8859-1', na_values=['NO_CONTENT','no-content'])  # header=0 is removed as the feature files has the headers by mistake!

    logger.info('loading training data ...')
    train = pd.read_csv('../../../dataset/hossein_sample_train.txt', names=['description_id', 'paper_id', 'description_text'], encoding='ISO-8859-1')  # header=0 is removed as the feature files has the headers by mistake!

    random_sample_idx = [i for i in range(len(train))]

    feature_types = ['bm.1_1_gram', 'bm.1_2_gram']
    # feature_types += ['tc.1_1_gram', 'tc.1_2_gram'] #it's commented for the vector is not normalized!
    feature_types += ['tf.1_1_gram', 'tf.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    feature_types += ['emd.50', 'emd.100', 'emd.200', 'emd.300']
    feature_types += ['glv.emd.50', 'glv.emd.100', 'glv.emd.200', 'glv.emd.300']
    feature_types += ['d2v.50', 'd2v.100', 'd2v.200', 'd2v.300']

    paper_info = ['title', 'abstract', 'journal']

    k = 2  # top-mapk most similar paper_[info] to description based on cosine similarity of feature vectors
    for info in paper_info:
        for feature_type in feature_types:
            topk(path + 'similarities/train.cos.{}.{}.{}'.format(info, feature_type, 3),
                 train,
                 papers,
                 path + 'topk/train.cos.{}.{}.{}.top{}'.format(info, feature_type, 3, k),
                 random_sample_idx,
                 k)


if __name__ == '__main__':
    test()
    exit(0)
    # path = 'D:/output/'
    path = '/mnt/sata_disk/hossein/wsdmcup2020/output/'

    logger.info('loading papers ...')
    papers = pd.read_csv('../../../dataset/papers.csv', names=['abstract', 'journal', 'keywords', 'paper_id', 'title', 'year'], encoding='ISO-8859-1', na_values=['NO_CONTENT', 'no-content'])#header=0 is removed as the feature files has the headers by mistake!

    logger.info('loading training data ...')
    train = pd.read_csv('../../../dataset/train.csv', names=['description_id', 'paper_id', 'description_text'], encoding='ISO-8859-1')#header=0 is removed as the feature files has the headers by mistake!

    random_sample_size = 1000
    logger.info('loading sampling indices {} ...'.format(random_sample_size))
    random_sample_idx = pd.read_csv(path + 'similarities/random/sampleindex.{}'.format(random_sample_size), header=None)
    random_sample_idx = [int(r) for i, r in random_sample_idx.iterrows()]

    feature_types = ['bm.1_1_gram', 'bm.1_2_gram']
    # feature_types += ['tc.1_1_gram', 'tc.1_2_gram'] #it's commented for the vector is not normalized!
    feature_types += ['tf.1_1_gram', 'tf.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    feature_types += ['emd.50', 'emd.100', 'emd.200', 'emd.300']
    feature_types += ['glv.emd.50', 'glv.emd.100', 'glv.emd.200', 'glv.emd.300']
    feature_types += ['d2v.50', 'd2v.100', 'd2v.200', 'd2v.300']

    paper_info = ['title', 'abstract', 'journal']

    k = 100  #top-mapk most similar paper_[info] to description based on cosine similarity of feature vectors
    for i in range(1, 10):
        for info in paper_info:
            for feature_type in feature_types:
                topk(path + 'similarities/random/train.cos.{}.{}.{}.{}'.format(info, feature_type, random_sample_size, i),
                     train,
                     papers,
                     path + 'topk/random/train.cos.{}.{}.{}.top{}.{}'.format(info, feature_type, random_sample_size, k, i),
                     random_sample_idx,
                     k)
