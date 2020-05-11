import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('top_k_train')

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
        topks = topks.append({'paper_id': train.paper_id[random_sample_idx[i]],
                              'true_paper_idx':papers[papers.paper_id == train.paper_id[random_sample_idx[i]]].index[0],
                              'topk': ','.join(papers.paper_id[topk]),
                              'topk_paper_idx': ','.join(str(idx) for idx in topk),
                              'score':','.join(str(s) for s in scores[i, :k])
                              }, ignore_index=True)
    logger.info('saving top-{} relevant papers for {} ...'.format(k, similarity_file))
    topks.to_csv(output, index=False)

if __name__ == '__main__':
    # batch_no = 1
    batch_no = int(sys.argv[1])
    path = '../output/'

    logger.info('loading papers ...')
    papers = pd.read_csv('../../../dataset/papers.csv', names=['abstract', 'journal', 'keywords', 'paper_id', 'title', 'year'], encoding='ISO-8859-1', na_values=['NO_CONTENT', 'no-content'])#header=0 is removed as the feature files has the headers by mistake!

    logger.info('loading training data ...')
    train = pd.read_csv('../../../dataset/train.csv', names=['description_id', 'paper_id', 'description_text'], encoding='ISO-8859-1')#header=0 is removed as the feature files has the headers by mistake!

    bath_size = 1000
    batch_idx = [i for i in range(bath_size * (batch_no - 1), np.minimum(bath_size * (batch_no), len(train)))]
    #
    # feature_types = ['bm.1_1_gram', 'bm.1_2_gram']
    # feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    #
    # paper_info = ['abstract','title']
    #
    k = 1000  #top-mapk most similar paper_[info] to description based on cosine similarity of feature vectors
    # for info in paper_info:
    #     for feature_type in feature_types:
    #         topk(path + 'similarities/train.cos.{}.{}.{}.{}'.format(info, feature_type, bath_size, batch_no),
    #              train,
    #              papers,
    #              path + 'topk/train.cos.{}.{}.{}.top{}.{}.v2'.format(info, feature_type, bath_size, k, batch_no),
    #              batch_idx,
    #              k)


    topk('../output/similarities/train.bm25.{}.{}'.format(bath_size, batch_no),
         train,
         papers,
         '../output/topk/train.bm25.{}.top{}.{}.v2'.format(bath_size, k, batch_no),
         batch_idx,
         k)



