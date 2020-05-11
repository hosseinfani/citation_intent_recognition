import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('vector_sim')

from sklearn.externals import joblib
from sklearn import metrics
import pandas as pd
import numpy as np

def test():
    path = '../output/'
    feature_types = ['bm.1_1_gram', 'bm.1_2_gram']
    # feature_types += ['tc.1_1_gram', 'tc.1_2_gram'] #it's commented for the vector is not normalized!
    feature_types += ['tf.1_1_gram', 'tf.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    feature_types += ['emd.50', 'emd.100', 'emd.200', 'emd.300']
    feature_types += ['glv.emd.50', 'glv.emd.100', 'glv.emd.200', 'glv.emd.300']
    feature_types += ['d2v.50', 'd2v.100', 'd2v.200', 'd2v.300']

    paper_info = ['title', 'abstract', 'journal']

    logger.info('loading papers ...')
    papers = pd.read_csv('../../../dataset/hossein_sample_papers.txt', names=['abstract', 'journal', 'keywords', 'paper_id', 'title', 'year'], encoding='ISO-8859-1',na_values=['NO_CONTENT', 'no-content'])

    logger.info('loading training data ...')
    train = pd.read_csv('../../../dataset/hossein_sample_train.txt', names=['description_id', 'paper_id', 'description_text'], encoding='ISO-8859-1')

    random_sample_idx = [i for i in range(len(train))]

    columns = []
    for info in paper_info:
        for feature_type in feature_types:
            columns.append('{}.{}'.format(info, feature_type))

    sims = pd.DataFrame(columns=columns)

    for info in paper_info:
        for feature_type in feature_types:
            train_feature_file = path + 'features/train.{}'.format(feature_type)
            paper_feature_file = path + 'features/{}.{}'.format(info, feature_type)
            logger.info('loading feature files {} and {} ...'.format(train_feature_file, paper_feature_file))
            train_features = joblib.load(train_feature_file)
            paper_features = joblib.load(paper_feature_file)

            logger.info('calculating true description-paper cosine similarity ...')
            for i in random_sample_idx:
                paper_idx = papers[papers.paper_id == train.paper_id[i]].index[0]
                score = metrics.pairwise.cosine_similarity(X=train_features[i, :].reshape(1, -1),
                                                          Y=paper_features[paper_idx, :].reshape(1, -1))[0][0]
                if len(sims) < len(random_sample_idx):
                    sims = sims.append({'{}.{}'.format(info, feature_type): score}, ignore_index=True)
                else:
                    sims.loc[i, '{}.{}'.format(info, feature_type)] = score

    sims.to_csv(path + 'similarities/truthsimilarity.{}'.format(len(train)), index=False)


if __name__ == '__main__':
    # test()
    # exit(0)
    # path = 'D:/output/'
    path = '/mnt/sata_disk/hossein/wsdmcup2020/output/'

    feature_types  = ['bm.1_1_gram', 'bm.1_2_gram']
    # feature_types += ['tc.1_1_gram', 'tc.1_2_gram'] #it's commented for the vector is not normalized!
    feature_types += ['tf.1_1_gram', 'tf.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    feature_types += ['emd.50', 'emd.100', 'emd.200', 'emd.300']
    feature_types += ['glv.emd.50', 'glv.emd.100', 'glv.emd.200', 'glv.emd.300']
    feature_types += ['d2v.50', 'd2v.100', 'd2v.200', 'd2v.300']

    paper_info = ['title', 'abstract', 'journal']

    logger.info('loading papers ...')
    papers = pd.read_csv('../../../dataset/papers.csv', names=['abstract', 'journal', 'keywords', 'paper_id', 'title', 'year'], encoding='ISO-8859-1', na_values=['NO_CONTENT', 'no-content'])# header=0 is removed as the feature files has the headers by mistake!

    logger.info('loading training data ...')
    train = pd.read_csv('../../../dataset/train.csv', names=['description_id', 'paper_id', 'description_text'], encoding='ISO-8859-1')# header=0 is removed as the feature files has the headers by mistake!

    random_sample_size = 1000
    logger.info('loading sampling indices {} ...'.format(random_sample_size))
    random_sample_idx = pd.read_csv(path + 'similarities/sampleindex.{}'.format(random_sample_size), header=None)
    random_sample_idx = [int(r) for i, r in random_sample_idx.iterrows()]

    columns = []
    for info in paper_info:
        for feature_type in feature_types:
            columns.append('{}.{}'.format(info, feature_type))

    sims = pd.DataFrame(columns=columns)

    for info in paper_info:
        for feature_type in feature_types:
            train_feature_file = path + 'features/train.{}'.format(feature_type)
            paper_feature_file = path + 'features/{}.{}'.format(info, feature_type)
            logger.info('loading feature files {} and {} ...'.format(train_feature_file, paper_feature_file))
            train_features = joblib.load(train_feature_file)
            paper_features = joblib.load(paper_feature_file)

            logger.info('calculating true description-paper cosine similarity ...')
            for i, idx in enumerate(random_sample_idx):
                paper_idx = papers[papers.paper_id == train.paper_id[idx]].index[0]
                score = metrics.pairwise.cosine_similarity(X=train_features[idx, :].reshape(1,-1), Y=paper_features[paper_idx,:].reshape(1,-1))[0][0]
                if len(sims) < len(random_sample_idx):
                    sims = sims.append({'{}.{}'.format(info, feature_type): score}, ignore_index=True)
                else:
                    sims.loc[i, '{}.{}'.format(info, feature_type)] = score

    sims.to_csv(path + 'similarities/truthsimilarity.{}'.format(random_sample_size), index=False)
