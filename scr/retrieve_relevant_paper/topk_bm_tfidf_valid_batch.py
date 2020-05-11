import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('top_k_valid')

import pandas as pd
from sklearn.externals import joblib
from sklearn import metrics
import numpy as np

def topk(similarity_file, valid, papers, output, sample_idx, k=10):
    logger.info('loading similarity file {} ...'.format(similarity_file))
    sims = joblib.load(similarity_file)

    logger.info('finding top-{} papers for {} ...'.format(k, similarity_file))
    ind = np.argsort(sims, axis=1)[:, :-(k+1):-1]
    scores = np.take_along_axis(sims, ind, axis=1)
    topks = pd.DataFrame()
    for i, topk in enumerate(ind[:, :k]):
        topks = topks.append({'description_id': valid.description_id[sample_idx[i]], 'topk': ','.join(papers.paper_id[topk]), 'score':','.join(str(s) for s in scores[i, :k])}, ignore_index=True)

    logger.info('saving top-{} relevant papers for {} ...'.format(k, similarity_file))
    topks.to_csv(output, index=False)

if __name__ == '__main__':
    batch_no = int(sys.argv[1])
    path = '/home/fattane/wsdmcup2020/collaborators/hossein/output/'

    logger.info('loading papers ...')
    papers = pd.read_csv('../../../dataset/papers.csv', names=['abstract', 'journal', 'keywords', 'paper_id', 'title', 'year'], encoding='ISO-8859-1', na_values=['NO_CONTENT', 'no-content'])#header=0 is removed as the feature files has the headers by mistake!

    logger.info('loading validation data ...')
    valid = pd.read_csv('../../../dataset/validation.csv', names=['description_id', 'description_text'], encoding='ISO-8859-1')#header=0 is removed as the feature files has the headers by mistake!

    bath_size = 1000
    batch_idx = [i for i in range(bath_size * (batch_no - 1), np.minimum(bath_size * (batch_no), len(valid)))]

    feature_types = ['bm.1_1_gram', 'bm.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']

    paper_info = ['title', 'abstract']

    k = 1000  #top-mapk most similar paper_[info] to description based on cosine similarity of feature vectors
    for info in paper_info:
        for feature_type in feature_types:
            topk(path + 'similarities/valid.cos.{}.{}.{}.{}'.format(info, feature_type, bath_size, batch_no),
                 valid,
                 papers,
                 path + 'topk/valid.cos.{}.{}.{}.top{}.{}'.format(info, feature_type, bath_size, k, batch_no),
                 batch_idx,
                 k)
