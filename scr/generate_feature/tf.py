import logging, sys
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('tf')

import gensim as gs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from sklearn.externals import joblib

def tf(train_file, validation_file, papers_file, ngrams, output_folder):

    logger.info('loading training data ...')
    train = pd.read_csv(train_file, names=['description_id', 'paper_id', 'description_text'], encoding='ISO-8859-1')#, header=0)#.sample(n=1000)
    n_train = len(train)
    logger.info('{} training data has been loaded.'.format(n_train))

    logger.info('loading validation data ...')
    valid = pd.read_csv(validation_file, names=['description_id', 'description_text'], encoding='ISO-8859-1')#, header=0)#.sample(n=1000)
    n_valid = len(valid)
    logger.info('{} validation data has been loaded.'.format(n_valid))

    logger.info('loading papers ...')
    papers = pd.read_csv(papers_file, names=['abstract','journal','keywords','paper_id','title','year'],encoding='ISO-8859-1', na_values=['NO_CONTENT','no-content'])#, header=0)#.sample(n=1000)
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

    for ngram in ngrams:
        logger.info('building {}-gram tc matrix ...'.format(ngram))
        cv = CountVectorizer(ngram_range=ngram)
        tcs = cv.fit_transform(pd.concat([train['processed_description'],
                                              valid['processed_description'],
                                              papers['processed_abstract'],
                                              papers['processed_journal'],
                                              papers['processed_title']]))
        joblib.dump(cv.vocabulary_, output_folder + 'vocabs')

        logger.info('saving tcs ...')
        joblib.dump(tcs[0: n_train], output_folder + 'train.tc.{}_{}_gram'.format(ngram[0],ngram[1]))
        joblib.dump(tcs[n_train: n_train + n_valid], output_folder + 'valid.tc.{}_{}_gram'.format(ngram[0],ngram[1]))
        joblib.dump(tcs[n_train + n_valid: n_train + n_valid + n_papers], output_folder + 'abstract.tc.{}_{}_gram'.format(ngram[0],ngram[1]))
        joblib.dump(tcs[n_train + n_valid + n_papers: n_train + n_valid + 2 * n_papers], output_folder + 'journal.tc.{}_{}_gram'.format(ngram[0],ngram[1]))
        joblib.dump(tcs[n_train + n_valid + 2 * n_papers:], output_folder + 'title.tc.{}_{}_gram'.format(ngram[0],ngram[1]))

        logger.info('building {}-gram tf matrix ...'.format(ngram))
        cv = TfidfVectorizer(use_idf=False, ngram_range=ngram)
        tfs = cv.fit_transform(pd.concat([train['processed_description'],
                                              valid['processed_description'],
                                              papers['processed_abstract'],
                                              papers['processed_journal'],
                                              papers['processed_title']]))
        joblib.dump(cv.vocabulary_, output_folder + 'vocabs')
        logger.info('saving tfs...')
        joblib.dump(tfs[0: n_train], output_folder + 'train.tf.{}_{}_gram'.format(ngram[0],ngram[1]))
        joblib.dump(tfs[n_train: n_train + n_valid], output_folder + 'valid.tf.{}_{}_gram'.format(ngram[0],ngram[1]))
        joblib.dump(tfs[n_train + n_valid: n_train + n_valid + n_papers], output_folder + 'abstract.tf.{}_{}_gram'.format(ngram[0],ngram[1]))
        joblib.dump(tfs[n_train + n_valid + n_papers: n_train + n_valid + 2 * n_papers], output_folder + 'journal.tf.{}_{}_gram'.format(ngram[0],ngram[1]))
        joblib.dump(tfs[n_train + n_valid + 2 * n_papers:], output_folder + 'title.tf.{}_{}_gram'.format(ngram[0],ngram[1]))

def test():
    tf(train_file="../../../dataset/hossein_sample_train.txt",
       validation_file="../../../dataset/hossein_sample_valid.txt",
       papers_file="../../../dataset/hossein_sample_papers.txt",
       ngrams=[(1, 1), (1, 2)],
       output_folder="../output/features/")

if __name__ == '__main__':
    test()
    exit(0)
    # path = 'D:/output/'
    path = '/mnt/sata_disk/hossein/wsdmcup2020/output/'

    tf(train_file="../../../dataset/train.csv",
       validation_file="../../../dataset/validation.csv",
       papers_file="../../../dataset/papers.csv",
       ngrams=[(1,1), (1,2)],
       output_folder= path + "features/")


