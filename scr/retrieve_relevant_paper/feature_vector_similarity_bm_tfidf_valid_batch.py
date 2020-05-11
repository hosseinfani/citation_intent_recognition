import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('vector_sim_valid')

from sklearn.externals import joblib
from sklearn import metrics
import pandas as pd
import numpy as np

def vector_sim(desc_feature_file, paper_feature_file, metric, output, random_sample_idx):
    logger.info('loading feature files {} and {} ...'.format(desc_feature_file, paper_feature_file))
    desc_features = joblib.load(desc_feature_file)
    paper_features = joblib.load(paper_feature_file)

    logger.info('calculating pairwise {} similarities ...'.format(metric))
    sims = 1 - metrics.pairwise_distances(X=desc_features[random_sample_idx,:], Y=paper_features, metric=metric, n_jobs=-1)

    logger.info('saving pairwise {} similarities ...'.format(metric))
    joblib.dump(sims, output, compress=3)

if __name__ == '__main__':
    batch_no = int(sys.argv[1])

    path = '/home/fattane/wsdmcup2020/collaborators/hossein/output/'

    feature_types  = ['bm.1_1_gram', 'bm.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    paper_info = ['title', 'abstract']

    valid = pd.read_csv('../../../dataset/validation.csv', names=['description_id', 'description_text'], encoding='ISO-8859-1')# header=0 is removed as the feature files has the headers by mistake!
    bath_size = 1000
    batch_idx = [i for i in range(bath_size * (batch_no - 1), np.minimum(bath_size * (batch_no), len(valid)))]

    logger.info('calculating pairwise cosine similarity for description batch {} from {} to {} ...'.format(batch_no, batch_idx[0], batch_idx[-1]))
    for info in paper_info:
        for feature_type in feature_types:
            vector_sim(path + 'features/valid.{}'.format(feature_type),
                       path + 'features/{}.{}'.format(info, feature_type),
                       'cosine',
                       path + 'similarities/valid.cos.{}.{}.{}.{}'.format(info, feature_type, bath_size, batch_no),
                       batch_idx)
