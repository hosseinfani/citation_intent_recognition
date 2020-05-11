import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('bm25_train')

from sklearn.externals import joblib
from sklearn import metrics
import pandas as pd
import numpy as np
import gensim as gs

if __name__ == '__main__':
    batch_no = int(sys.argv[1])
    # batch_no = 1

    path = '/data2/hossein/wsdmcup2020/collaborators/hossein/output/'

    logger.info('loading training data ...')
    train = pd.read_csv('../../../dataset/train.csv', names=['description_id', 'paper_id', 'description_text'], encoding='ISO-8859-1')#.sample(100)# header=0 is removed as the feature files has the headers by mistake!
    bath_size = 1000
    batch_idx = [bath_size * (batch_no - 1), np.minimum(bath_size * (batch_no), len(train))]

    logger.info('loading validation data ...')
    valid = pd.read_csv('../../../dataset/validation.csv', names=['description_id', 'description_text'],encoding='ISO-8859-1')#.sample(100)  # , header=0)#.sample(n=100)

    logger.info('loading papers ...')
    papers = pd.read_csv('../../../dataset/papers.csv', names=['abstract', 'journal', 'keywords', 'paper_id', 'title', 'year'],encoding='ISO-8859-1', na_values=['NO_CONTENT', 'no-content'])#.sample(100)  # , header=0)#.sample(n=100)
    n_papers = len(papers)

    preprocessing_funcs = [gs.parsing.strip_tags,
                           gs.parsing.strip_punctuation,
                           gs.parsing.strip_multiple_whitespaces,
                           gs.parsing.strip_numeric,
                           gs.parsing.remove_stopwords,
                           gs.parsing.strip_short,
                           gs.parsing.strip_multiple_whitespaces,
                           gs.parsing.stem_text]

    logger.info('preprocessig on texts ...')
    train['processed_description'] = train.apply(lambda r: ' '.join(gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs, s=r['description_text'].lower())),
        deacc=True, min_len=3, max_len=15)), axis=1)
    valid['processed_description'] = valid.apply(lambda r: ' '.join(gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs, s=r['description_text'].lower())),
        deacc=True, min_len=3, max_len=15)), axis=1)
    papers['processed_all'] = papers.apply(lambda r: ' '.join(gs.utils.simple_preprocess(' '.join(gs.parsing.preprocess_string(filters=preprocessing_funcs,
                                    s=(r['abstract'] if not pd.isnull(r['abstract']) else ' NaN ')+(r['title'])+(r['journal']if not pd.isnull(r['journal']) else ' NaN ').lower())), deacc=True,
        min_len=0, max_len=50)), axis=1)

    texts = pd.concat([papers['processed_all'], train['processed_description'],valid['processed_description']])

    logger.info('building corpus ...')
    texts_corpus = [[word for word in text.split()] for text in texts.tolist()]
    train_corpus = [[word for word in text.split()] for text in train['processed_description'][batch_idx[0]:batch_idx[1]].tolist()]

    logger.info('calculating bm scores for the papers ...')
    bm = gs.summarization.bm25.BM25(texts_corpus)
    logger.info('calculating bm scores for the train-papers ...')
    train_scores = [bm.get_scores(text)[:n_papers] for text in train_corpus]


    logger.info('saving bm scores ...')
    joblib.dump(np.asarray(train_scores), '../output/similarities/train.bm25.{}.{}'.format(bath_size, batch_no), compress=3)

