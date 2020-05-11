import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('w2v_mean')

import gensim as gs
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
import re
from sklearn.externals import joblib

def mean_w2v(train_file, validation_file, papers_file, output_folder, dims=None, pretraineds=None):
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

    if pretraineds:
        for pretrained in pretraineds:
            logger.info('loading pretrained model {} ...'.format(pretrained))
            dim = int(pretrained.split('.')[-2].replace('d', ''))
            with open(pretrained, encoding="utf8") as lines:
                w2v = {line.split()[0]: np.array(list(map(float, line.split()[1:]))) for line in lines}#[next(lines) for x in range(1000)]}
            logger.info('saving mean embedding of size {} using pretrained model {} ...'.format(dim, pretrained))
            joblib.dump(_mean_w2v(train['description_text'], w2v, dim, pretrained=True), output_folder + 'train.glv.emd.{}'.format(dim))
            joblib.dump(_mean_w2v(valid['description_text'], w2v, dim, pretrained=True), output_folder + 'valid.glv.emd.{}'.format(dim))
            joblib.dump(_mean_w2v(papers['abstract'], w2v, dim, pretrained=True), output_folder + 'abstract.glv.emd.{}'.format(dim))
            joblib.dump(_mean_w2v(papers['journal'], w2v, dim, pretrained=True), output_folder + 'journal.glv.emd.{}'.format(dim))
            joblib.dump(_mean_w2v(papers['title'], w2v, dim, pretrained=True), output_folder + 'title.glv.emd.{}'.format(dim))
        return

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

    for dim in dims:
        w2v = Word2Vec(sentences=texts, size=dim, iter=100, workers=multiprocessing.cpu_count())
        logger.info('saving mean embedding of size {} ...'.format(dim))
        joblib.dump(_mean_w2v(train['processed_description'], w2v.wv, dim), output_folder + 'train.emd.{}'.format(dim))
        joblib.dump(_mean_w2v(valid['processed_description'], w2v.wv, dim), output_folder + 'valid.emd.{}'.format(dim))
        joblib.dump(_mean_w2v(papers['processed_abstract'], w2v.wv, dim), output_folder + 'abstract.emd.{}'.format(dim))
        joblib.dump(_mean_w2v(papers['processed_journal'], w2v.wv, dim), output_folder + 'journal.emd.{}'.format(dim))
        joblib.dump(_mean_w2v(papers['processed_title'], w2v.wv, dim), output_folder + 'title.emd.{}'.format(dim))

def _mean_w2v(texts, w2v, dim, pretrained=False):
    res = np.zeros((len(texts.tolist()), dim))
    for i, text in enumerate(texts):
        text = re.sub("[^a-zA-Z]", " ", text if not pd.isna(text) else '') if pretrained else ' '.join(text)
        words = text.lower().split()
        res[i] = mean(words, w2v, dim)
    return res

def mean(words, w2v, dim):
        nwords = 0.
        embedding = np.zeros((dim,), dtype="float32")
        for word in words:
            if word in w2v:
                nwords = nwords + 1.
                embedding = np.add(embedding, w2v[word])
        return  np.divide(embedding, nwords) if nwords > 0 else embedding

def test():
    mean_w2v(train_file="../../../dataset/hossein_sample_train.txt",
             validation_file="../../../dataset/hossein_sample_valid.txt",
             papers_file="../../../dataset/hossein_sample_papers.txt",
             output_folder="../output/features/",
             dims=[50, 100, 200, 300])
    mean_w2v(train_file="../../../dataset/hossein_sample_train.txt",
             validation_file="../../../dataset/hossein_sample_valid.txt",
             papers_file="../../../dataset/hossein_sample_papers.txt",
             output_folder="../output/features/",
             pretraineds=['../../../dataset/pretrained/glove.6B.50d.txt',
                          '../../../dataset/pretrained/glove.6B.100d.txt',
                          '../../../dataset/pretrained/glove.6B.200d.txt',
                          '../../../dataset/pretrained/glove.6B.300d.txt'])

if __name__ == '__main__':
    test()
    exit(0)
    # path = 'D:/output/'
    path = '/mnt/sata_disk/hossein/wsdmcup2020/output/'
    mean_w2v(train_file="../../../dataset/train.csv",
              validation_file="../../../dataset/validation.csv",
              papers_file="../../../dataset/papers.csv",
              output_folder=path + "features/",
              dims=[50, 100, 200, 300])

    mean_w2v(train_file="../../../dataset/train.csv",
              validation_file="../../../dataset/validation.csv",
              papers_file="../../../dataset/papers.csv",
              output_folder=path + "features/",
              pretraineds=['../../../dataset/pretrained/glove.6B.50d.txt',
                          '../../../dataset/pretrained/glove.6B.100d.txt',
                          '../../../dataset/pretrained/glove.6B.200d.txt',
                          '../../../dataset/pretrained/glove.6B.300d.txt'])