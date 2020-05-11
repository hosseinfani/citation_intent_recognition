import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy import sparse
import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('classifier')

sys.path.extend(['./evaluation'])
from eval import ap_k

random_seed = 7881
np.random.seed(random_seed)



def load_train(retrieval_method, n, topk, path):
    logger.info('loading papers ...')
    papers = pd.read_csv('../../../dataset/papers.csv',names=['abstract', 'journal', 'keywords', 'paper_id', 'title', 'year'], encoding='ISO-8859-1',na_values=['NO_CONTENT', 'no-content'])  # , header=0)#.sample(n=100)

    logger.info('loading training data ...')
    train = pd.read_csv('../../../dataset/train.csv', names=['description_id', 'paper_id', 'description_text'],encoding='ISO-8859-1')  # , header=0)#.sample(n=100)
    train = train[:n]

    batch_size = 1000

    #fetch topk relevant papers for all training instances
    logger.info('loading top{} papers for training instances for {} ...'.format(topk, retrieval_method))
    topk_train = pd.DataFrame(columns=['paper_id', 'score', 'topk', 'true_paper_idx', 'topk_paper_idx'])

    for batch_no in range(1, (int(len(train) / batch_size) + 1) + 1):
        logger.info('loading training batch {} for {} ...'.format(batch_no, retrieval_method))
        top1000 = pd.read_csv(path + 'topk/train.cos.{}.{}.top{}.{}'.format(retrieval_method, batch_size, topk, batch_no), header=0)
        topk_train = topk_train.append(top1000, ignore_index=True, sort=False)
    topk_train = topk_train[:n]

    paper_id_2_idx = dict(zip(papers.paper_id, papers.index))

    #find the index of topk papers and true paper in the papers corpus; needed to find respective paper's features
    #in the *.v2 versions of files, it's already there
    logger.info('finding papers indices ...')
    for idx, r in topk_train.iterrows():
        topk_train.loc[idx]['true_paper_idx'] = paper_id_2_idx[str(r.paper_id)]
        topk_paper_ids = r.topk.replace('paper_id,', '').split(',')
        topk_train.loc[idx]['topk_paper_idx'] = [paper_id_2_idx[str(x)] for x in topk_paper_ids[:topk]]
        topk_train.loc[idx]['topk'] = topk_paper_ids[:topk]
    return papers, train, topk_train

def prepare_X_y(paper_features, train_features, topk_train, fold_idx, topk_4_train_test):
    X = []; y = [];paper_ids = []
    for idx, r in (topk_train.iloc[fold_idx]).iterrows():
        # (description, true_paper, 1) instance
        paper_ids.append(r.paper_id)
        true_paper_features = paper_features[topk_train.loc[idx]['true_paper_idx'], :]
        X.append(sparse.hstack([train_features[idx, :], true_paper_features]))
        y.append(1)

        # (description, top_paper, 2) instances
        for i, p_idx in enumerate(topk_train.loc[idx]['topk_paper_idx'][:topk_4_train_test]):
            topk_features = paper_features[p_idx, :]
            X.append(sparse.hstack([train_features[idx, :], topk_features]))
            if topk_train.loc[idx]['topk'][i] == r.paper_id:
                y.append(1)
            else:
                y.append(2)
        [paper_ids.append(paper_id) for paper_id in topk_train.loc[idx]['topk']]
    return sparse.vstack(X).tocsr(), y, paper_ids

def load_test(retrieval_method, n, k, path):
    logger.info('loading papers ...')
    papers = pd.read_csv('../../../dataset/papers.csv',names=['abstract', 'journal', 'keywords', 'paper_id', 'title', 'year'], encoding='ISO-8859-1',na_values=['NO_CONTENT', 'no-content'])  # , header=0)#.sample(n=100)

    logger.info('loading test data ...')
    test = pd.read_csv('../../../dataset/validation.csv', names=['description_id', 'description_text'],encoding='ISO-8859-1')  # , header=0)#.sample(n=100)
    test = test[:n]

    batch_size = 1000
    topk = 1000

    #fetch topk relevant papers for all training instances
    logger.info('loading top{} papers for test instances for {} ...'.format(topk, retrieval_method))
    topk_test = pd.DataFrame(columns=['description_id', 'score', 'topk', 'topk_paper_idx'])

    for batch_no in range(1, (int(len(test) / batch_size) + 1) + 1):
        logger.info('loading test batch {} for {} ...'.format(batch_no, retrieval_method))
        top1000 = pd.read_csv(path + 'topk/valid.cos.{}.{}.top{}.{}'.format(retrieval_method, batch_size, topk, batch_no), header=0)
        topk_test = topk_test.append(top1000, ignore_index=True, sort=False)
    topk_test = topk_test[:n]

    paper_id_2_idx = dict(zip(papers.paper_id, papers.index))

    #find the index of topk papers and true paper in the papers corpus; needed to find respective paper's features
    #in the *.v2 versions of files, it's already there
    logger.info('finding papers indices ...')
    for idx, r in topk_test.iterrows():
        topk_paper_ids = r.topk.replace('paper_id,', '').split(',')#remove the true paper from topk papers
        topk_test.loc[idx]['topk_paper_idx'] = [paper_id_2_idx[str(x)] for x in topk_paper_ids[:k]]
        topk_test.loc[idx]['topk'] = topk_paper_ids[:k]
    return papers, test, topk_test

def train(feature_types, path, retrieval_method, n, topk, topk_4_train):
    #todo: replace with mixed version
    papers, train, topk_train = load_train(retrieval_method=retrieval_method, n=n, topk=topk, path='../output/')
    n_train = len(train)
    n_papers = len(papers)

    paper_info = ['abstract','title']
    #todo: add year feature

    model = RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=10, random_state=random_seed)

    #todo: complete the feature set: 1) add cos score, 2) bm25 score,
    for feature_type in feature_types:
        train_features = joblib.load(path + 'features/train.{}'.format(feature_type))
        paper_features = []
        for info in paper_info:
            paper_features.append(joblib.load(path + 'features/{}.{}'.format(info, feature_type)))
        paper_features = sparse.hstack(paper_features).tocsr()

        logger.info('model x-val for feat-{}.{}.ret-{}.n{}.top{}.train-top{}.fold1.{} ...'.format(feature_type ,'.'.join(paper_info), retrieval_method, n, topk, topk_4_train, type(model).__name__))
        logger.info('training for fold{} ...'.format(1))
        X = [];y = [];
        for idx, r in topk_train.iterrows():
            true_paper_features = paper_features[topk_train.loc[idx]['true_paper_idx'], :]
            X.append(sparse.hstack([train_features[idx, :], true_paper_features]))
            y.append(1)
            for p_idx in topk_train.loc[idx]['topk_paper_idx'][:topk_4_train]:
                topk_features = paper_features[p_idx, :]
                X.append(sparse.hstack([train_features[idx, :], topk_features]))
                y.append(2)
        X = sparse.vstack(X).tocsr()
        model.fit(X, y)
        joblib.dump(model, path + 'models/feat-{}.{}.ret-{}.n{}.top{}.train-top{}.fold1.{}.f0'.format(feature_type ,'.'.join(paper_info), retrieval_method, n, topk, topk_4_train, type(model).__name__))

def kfold_x_valid(feature_types, path, retrieval_method, n, topk, folds, mapk, topk_4_train, topk_4_test):
    #todo: replace with mixed version
    papers, train, topk_train = load_train(retrieval_method=retrieval_method, n=n, topk=topk, path='../output/')
    n_train = len(train)
    n_papers = len(papers)

    paper_info = ['abstract','title']
    #todo: add year feature

    skf = KFold(n_splits=folds, shuffle=True, random_state=random_seed)
    model = RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=10, random_state=random_seed)

    #todo: complete the feature set: 1) add cos score, 2) bm25 score,
    for feature_type in feature_types:
        train_features = joblib.load(path + 'features/train.{}'.format(feature_type))
        paper_features = []
        for info in paper_info:
            paper_features.append(joblib.load(path + 'features/{}.{}'.format(info, feature_type)))
        paper_features = sparse.hstack(paper_features).tocsr()

        kfold_map_k = []
        kfold_map_k_r = []
        logger.info('model x-val for feat-{}.{}.ret-{}.n{}.top{}.fold{}.{} ...'.format(feature_type ,'.'.join(paper_info), retrieval_method, n, topk, folds, type(model).__name__))
        for i, (train_fold_idx, test_fold_idx) in enumerate(skf.split(range(n_train))):
            logger.info('training for fold{} ...'.format(i))
            X, y, _ = prepare_X_y(paper_features, train_features, topk_train, train_fold_idx, topk_4_train)
            model.fit(X, y)
            joblib.dump(model, path + 'models/feat-{}.{}.ret-{}.n{}.top{}.train-top{}.fold{}.{}.f{}'.format(feature_type ,'.'.join(paper_info), retrieval_method, n, topk, topk_4_train, folds, type(model).__name__, i))

            logger.info('testing for fold{} ...'.format(i, info, feature_type))
            map_k = []
            map_k_r = []
            for idx, r in (topk_train.iloc[test_fold_idx]).iterrows():
                X = []; y = [];
                # if idx % 100 == 0:
                #     logger.info('test instance idx {} ...'.format(idx))
                X, _, paper_ids = prepare_X_y(paper_features, train_features, topk_train, [idx], topk_4_test)
                X = X[1:,:]#remove the (desc, true_paper) instance!!!
                y.append(paper_ids[0])
                paper_ids = paper_ids[1:]#remove the true_paper id
                prob = model.predict_proba(X)[:, 0]
                ind_asc = np.argsort(prob)
                ind_desc = ind_asc[: -(len(prob) + 1): -1]
                pred = np.asarray(paper_ids)[[ind_desc]]
                pred_r = np.asarray(paper_ids)[[ind_asc]]
                #todo: save the prediction for stacking
                map_k.append(ap_k(y, pred, mapk))
                map_k_r.append(ap_k(y, pred_r, mapk))
                # logger.info('{}:{}'.format(ap_k(y, pred, mapk), ap_k(y, pred_r, mapk)))

            kfold_map_k.append(np.mean(np.asarray(map_k)))
            kfold_map_k_r.append(np.mean(np.asarray(map_k_r)))
            logger.info('map@{}.fold{} for {}.{}: avg={}, avg_r={}'.format(mapk, i, info, feature_type, np.mean(np.asarray(map_k)), np.mean(np.asarray(map_k_r))))
        logger.info('map@{} all folds: avg={}, std={}'.format(mapk, np.mean(np.asarray(kfold_map_k)), np.std(np.asarray(kfold_map_k))))

def test(feature_types, path, retrieval_method, n, topk, folds, topk_4_train):
    papers, test, topk_test = load_test(retrieval_method=retrieval_method, n=n, k=topk, path='../output/')
    n_test = len(test)
    n_papers = len(papers)

    paper_info = ['abstract', 'title']
    # todo: add year feature

    # todo: complete the feature set: 1) add cos score, 2) bm25 score,
    for feature_type in feature_types:
        test_features = joblib.load(path + 'features/valid.{}'.format(feature_type))
        paper_features = []
        for info in paper_info:
            paper_features.append(joblib.load(path + 'features/{}.{}'.format(info, feature_type)))
        paper_features = sparse.hstack(paper_features).tocsr()

        for i in range(folds):
            logger.info('loading model for feat-{}.{}.ret-{}.n{}.top{}.train-top{}.fold{}.{}.f{} ...'.format(feature_type,'.'.join(paper_info),retrieval_method, 100000, 1000, topk_4_train, folds, 'RandomForestClassifier', i))
            model = joblib.load(path + 'models/feat-{}.{}.ret-{}.n{}.top{}.train-top{}.fold{}.{}.f{}'.format(feature_type,'.'.join(paper_info),retrieval_method, 100000, 1000, topk_4_train, folds, 'RandomForestClassifier', i))

            submission = pd.DataFrame(columns=['description_id', 'top1', 'top2', 'top3'])
            logger.info('testing for fold{} ...'.format(i, info, feature_type))
            for idx, r in topk_test.iterrows():
                if idx%100 == 0:
                    logger.info('test instance idx {} ...'.format(idx))
                X = []
                for p_idx in topk_test.loc[idx]['topk_paper_idx']:
                    topk_features = paper_features[p_idx, :]
                    X.append(sparse.hstack([test_features[idx, :], topk_features]))
                paper_ids = [paper_id for paper_id in topk_test.loc[idx]['topk']]
                X = sparse.vstack(X).tocsr()
                prob = model.predict_proba(X)[:, 0]
                ind = np.argsort(prob)[: -(len(prob) + 1): -1]
                pred = np.asarray(paper_ids)[[ind]]
                prob = prob[[ind]]
                submission = submission.append({'description_id':r.description_id,'top1':pred[0],'top2':pred[1],'top3':pred[2]}, ignore_index=True, sort=False)
            submission[1:].to_csv(path + '../submission/classifier/feat-{}.{}.ret-{}.n{}.top{}.train-top{}.fold{}.{}.f{}.csv'.format(feature_type, '.'.join(paper_info),retrieval_method, n, topk, topk_4_train, folds, type(model).__name__, i),
                                  columns=['description_id', 'top1', 'top2', 'top3'], index=False, header=None)


if __name__ == '__main__':
    feature_types = ['bm.1_1_gram']  # , 'bm.1_2_gram']
    # feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    # kfold_x_valid(feature_types=feature_types, path='../output/', retrieval_method='abstract.bm.1_1_gram', n=100000, topk=1000, folds=5, mapk=3, topk_4_train=10, topk_4_test=1000)
    kfold_x_valid(feature_types=feature_types, path='../output/', retrieval_method='abstract.bm.1_1_gram', n=100000, topk=1000, folds=5, mapk=3, topk_4_train=5, topk_4_test=10)
    # train(feature_types=feature_types, path='../output/', retrieval_method='abstract.bm.1_1_gram', n=100000, topk=10)

    # test(feature_types=feature_types, path='../output/', retrieval_method='abstract.bm.1_1_gram', n=100000, topk=6, folds=5, topk_4_train=3)

