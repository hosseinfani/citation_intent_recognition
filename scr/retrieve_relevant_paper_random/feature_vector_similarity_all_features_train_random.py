import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('vector_sim')

from sklearn.externals import joblib
from sklearn import metrics
import pandas as pd
import numpy as np

def vector_sim(train_feature_file, paper_feature_file, metric, output, random_sample_idx):
    logger.info('loading feature files {} and {} ...'.format(train_feature_file, paper_feature_file))
    train_features = joblib.load(train_feature_file)
    paper_features = joblib.load(paper_feature_file)

    logger.info('calculating pairwise {} similarities ...'.format(metric))
    sims = 1 - metrics.pairwise_distances(X=train_features[random_sample_idx,:], Y=paper_features, metric=metric, n_jobs=-1)

    logger.info('saving pairwise {} similarities ...'.format(metric))
    joblib.dump(sims, output, compress=3)

def test():
    path = '../output/'

    feature_types  = ['bm.1_1_gram', 'bm.1_2_gram']
    # feature_types += ['tc.1_1_gram', 'tc.1_2_gram'] #it's commented for the vector is not normalized!
    feature_types += ['tf.1_1_gram', 'tf.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    feature_types += ['emd.50', 'emd.100', 'emd.200', 'emd.300']
    feature_types += ['glv.emd.50', 'glv.emd.100', 'glv.emd.200', 'glv.emd.300']
    feature_types += ['d2v.50', 'd2v.100', 'd2v.200', 'd2v.300']

    paper_info = ['title', 'abstract', 'journal']
    train = pd.read_csv('../../../dataset/hossein_sample_train.txt', names=['description_id', 'paper_id', 'description_text'], encoding='ISO-8859-1')
    random_sample_idx = [i for i in range(0, len(train))]
    for info in paper_info:
        for feature_type in feature_types:
            vector_sim(path + 'features/train.{}'.format(feature_type),
                       path + 'features/{}.{}'.format(info, feature_type),
                       'cosine',
                       path + 'similarities/train.cos.{}.{}.{}'.format(info, feature_type, len(train)),
                       random_sample_idx)

if __name__ == '__main__':
    path = 'D:/output/'
    path = '/mnt/sata_disk/hossein/wsdmcup2020/output/'

    feature_types  = ['bm.1_1_gram', 'bm.1_2_gram']
    # feature_types += ['tc.1_1_gram', 'tc.1_2_gram'] #it's commented for the vector is not normalized!
    feature_types += ['tf.1_1_gram', 'tf.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    feature_types += ['emd.50', 'emd.100', 'emd.200', 'emd.300']
    feature_types += ['glv.emd.50', 'glv.emd.100', 'glv.emd.200', 'glv.emd.300']
    feature_types += ['d2v.50', 'd2v.100', 'd2v.200', 'd2v.300']

    paper_info = ['title', 'abstract', 'journal']
    random_sample_size = 1000
    for i in range(1,10):
        logger.info('sampling training data by {} for {} ...'.format(random_sample_size, i))
        train = pd.read_csv('../../../dataset/train.csv', names=['description_id', 'paper_id', 'description_text'], encoding='ISO-8859-1')# header=0 is removed as the feature files has the headers by mistake!
        random_sample_idx = np.random.randint(0, len(train), random_sample_size)
        with open(path + 'similarities/random/sampleindex.{}.{}'.format(random_sample_size, i), 'w') as f:
            np.savetxt(f, random_sample_idx, delimiter=',', fmt='%d')

        for info in paper_info:
            for feature_type in feature_types:
                vector_sim(path + 'features/train.{}'.format(feature_type),
                           path + 'features/{}.{}'.format(info, feature_type),
                           'cosine',
                           path + 'similarities/random/train.cos.{}.{}.{}.{}'.format(info, feature_type, random_sample_size, i),
                           random_sample_idx)
