import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('top3_valid')

def old_top3_most_similar_paper():
    path = '../output/topk/'
    feature_types = ['bm.1_1_gram', 'bm.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    paper_info = ['title', 'abstract']

    logger.info('loading validation data ...')
    # header=0 is removed as the feature files has the headers by mistake!
    valid = pd.read_csv('../../../dataset/validation.csv', names=['description_id', 'description_text'],encoding='ISO-8859-1')

    batch_size = 1000
    top_k = 1000
    for info in paper_info:
        for feature_type in feature_types:
            submission = pd.DataFrame()
            for batch_no in range(1, (int(len(valid)/batch_size) + 1) + 1):
                logger.info('loading validation batch {} for valid.cos.{}.{} ...'.format(batch_no, info, feature_type))
                top1000 = pd.read_csv(path + 'valid.cos.{}.{}.{}.top{}.{}'.format(info, feature_type, batch_size, top_k, batch_no), header=0)
                top1000['topk_new'] = top1000.apply(lambda r: r['topk'].replace('paper_id,', '').split(','), axis=1)
                top1000['top1'] = top1000.apply(lambda r: r['topk_new'][0], axis=1)
                top1000['top2'] = top1000.apply(lambda r: r['topk_new'][1], axis=1)
                top1000['top3'] = top1000.apply(lambda r: r['topk_new'][2], axis=1)
                df = valid.iloc[(batch_no - 1) * batch_size: batch_no * 1000]['description_id']
                df = df.reset_index()
                top1000['description_id'] = df.description_id
                submission = submission.append(top1000, ignore_index=True, sort=False)

            # submission.drop(submission[pd.isna(submission.description_id)].index, inplace=True) #controling line in submission sample file
            submission[1:].to_csv(path + '../../submission/old_top3_most_similar_paper/valid.cos.{}.{}.{}.top{}.csv'.format(info, feature_type, batch_size, top_k),columns=['description_id', 'top1', 'top2', 'top3'] , index=False, header=None)

def top3_most_similar_paper():
    path = '../output/topk/'
    feature_types = ['bm.1_1_gram', 'bm.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    paper_info = ['title', 'abstract']

    logger.info('loading validation data ...')
    # header=0 is removed as the feature files has the headers by mistake!
    valid = pd.read_csv('../../../dataset/validation.csv', names=['description_id', 'description_text'],encoding='ISO-8859-1')

    batch_size = 1000
    top_k = 1000
    for info in paper_info:
        for feature_type in feature_types:
            submission = pd.DataFrame(columns=['description_id', 'top1', 'top2', 'top3'])
            for batch_no in range(1, (int(len(valid) / batch_size) + 1) + 1):
                logger.info('loading validation batch {} for valid.cos.{}.{} ...'.format(batch_no, info, feature_type))
                top1000 = pd.read_csv(path + 'valid.cos.{}.{}.{}.top{}.{}'.format(info, feature_type, batch_size, top_k, batch_no), header=0)
                top1000['top1'] = top1000.apply(lambda r: r['topk'].replace('paper_id,', '').split(',')[0], axis=1)
                top1000['top2'] = top1000.apply(lambda r: r['topk'].replace('paper_id,', '').split(',')[1], axis=1)
                top1000['top3'] = top1000.apply(lambda r: r['topk'].replace('paper_id,', '').split(',')[2], axis=1)
                submission = submission.append(top1000, ignore_index=True, sort=False)
            submission[1:].to_csv(path + '../../submission/top3_most_similar_paper/valid.cos.{}.{}.{}.top{}.csv'.format(info, feature_type, batch_size, top_k),columns=['description_id', 'top1', 'top2', 'top3'], index=False, header=None)

def top3_most_similar_paper_excluding_training_papers():
    path = '../output/topk/'
    feature_types = ['bm.1_1_gram', 'bm.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']

    paper_info = ['title', 'abstract']

    # logger.info('loading papers ...')
    # papers = pd.read_csv('../../../dataset/papers.csv',names=['abstract', 'journal', 'keywords', 'paper_id', 'title', 'year'], encoding='ISO-8859-1',na_values=['NO_CONTENT','no-content'])  # header=0 is removed as the feature files has the headers by mistake!

    logger.info('loading validation data ...')
    # header=0 is removed as the feature files has the headers by mistake!
    valid = pd.read_csv('../../../dataset/validation.csv', names=['description_id', 'description_text'],encoding='ISO-8859-1')

    logger.info('loading training data ...')
    # header=0 is removed as the feature files has the headers by mistake!
    train = pd.read_csv('../../../dataset/train.csv', names=['description_id', 'paper_id', 'description_text'],encoding='ISO-8859-1')
    train_papers = set(train['paper_id'].tolist())


    batch_size = 1000
    top_k = 1000
    for info in paper_info:
        for feature_type in feature_types:
            submission = pd.DataFrame(columns=['description_id', 'top1', 'top2', 'top3'])
            for batch_no in range(1, (int(len(valid) / batch_size) + 1) + 1):
                logger.info('loading validation batch {} for valid.cos.{}.{} ...'.format(batch_no, info, feature_type))
                top1000 = pd.read_csv(path + 'valid.cos.{}.{}.{}.top{}.{}'.format(info, feature_type, batch_size, top_k, batch_no), header=0)
                top1000['topk'] = top1000.apply(lambda r: r['topk'].replace('paper_id,', '').split(','), axis=1)
                for idx, r in top1000.iterrows():
                    # dif of two list=> very slow
                    # dif of two sets=>fast but looses order
                    # dif of a list from a set=> very fast
                    # credit goes to Mark Byers at https://stackoverflow.com/questions/3462143/get-difference-between-two-lists
                    top1000.loc[idx]['topk'] = [p for p in r['topk'] if p not in train_papers]

                top1000['top1'] = top1000.apply(lambda r: r['topk'][0], axis=1)
                top1000['top2'] = top1000.apply(lambda r: r['topk'][1], axis=1)
                top1000['top3'] = top1000.apply(lambda r: r['topk'][2], axis=1)

                submission = submission.append(top1000, ignore_index=True, sort=False)

            submission[1:].to_csv(path + '../../submission/top3_most_similar_paper_excluding_training_papers/valid.cos.{}.{}.{}.top{}.csv'.format(info, feature_type, batch_size, top_k),columns=['description_id', 'top1', 'top2', 'top3'], index=False, header=None)

def top3_most_similar_paper_mixed():
    path = '../output/topk/'
    feature_types = ['bm.1_1_gram', 'bm.1_2_gram']
    # feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    paper_info = ['title', 'abstract']

    logger.info('loading validation data ...')
    # header=0 is removed as the feature files has the headers by mistake!
    valid = pd.read_csv('../../../dataset/validation.csv', names=['description_id', 'description_text'], encoding='ISO-8859-1')

    batch_size = 1000
    top_k = 1000
    combinations = []
    for info1 in paper_info:
        for feature_type1 in feature_types:
            for info2 in paper_info:
                for feature_type2 in feature_types:
                    if info1 == info2 and feature_type1 == feature_type2:
                        continue
                    if set(['{}.{}'.format(info1, feature_type1), '{}.{}'.format(info2, feature_type2)]) in combinations:
                        continue
                    combinations.append(set(['{}.{}'.format(info1, feature_type1), '{}.{}'.format(info2, feature_type2)]))
                    logger.info('valid.cos.{}.{} mixing with valid.cos.{}.{} ...'.format(info1,feature_type1,info2,feature_type2))
                    submission = pd.DataFrame(columns=['description_id', 'top1', 'top2', 'top3'])
                    for batch_no in range(1, (int(len(valid) / batch_size) + 1) + 1):
                        logger.info('loading validation batch {} ...'.format(batch_no))

                        topk_a = pd.read_csv(path + 'valid.cos.{}.{}.{}.top{}.{}'.format(info1, feature_type1, batch_size, top_k, batch_no), header=0)
                        topk_b = pd.read_csv(path + 'valid.cos.{}.{}.{}.top{}.{}'.format(info2, feature_type2, batch_size, top_k, batch_no), header=0)

                        for (idx_a, a), (idx_b, b) in zip(topk_a.iterrows(), topk_b.iterrows()):
                            merged_scores = [float(a) for a in a.score.split(',')] + [float(b) for b in b.score.split(',')]
                            merged_paper_ids = a.topk.split(',') + b.topk.split(',')
                            ind = np.argsort(merged_scores)[:-((2 * top_k) + 1): -1]
                            scores = np.asarray(merged_scores)[[ind]]
                            paper_ids = np.asarray(merged_paper_ids)[[ind]]

                            paper_ids_unique = []
                            scores_unique = []
                            for i, p in enumerate(paper_ids):
                                if p not in paper_ids_unique:
                                    paper_ids_unique.append(p)
                                    scores_unique.append(scores[i])

                            topk_a.loc[idx_a]['topk'] = ','.join(paper_ids_unique[:top_k])
                            topk_a.loc[idx_b]['score'] = ','.join(str(s) for s in scores_unique[:top_k])

                        topk_a['top1'] = topk_a.apply(lambda r: r['topk'].replace('paper_id,', '').split(',')[0], axis=1)
                        topk_a['top2'] = topk_a.apply(lambda r: r['topk'].replace('paper_id,', '').split(',')[1], axis=1)
                        topk_a['top3'] = topk_a.apply(lambda r: r['topk'].replace('paper_id,', '').split(',')[2], axis=1)
                        submission = submission.append(topk_a, ignore_index=True, sort=False)
                    submission[1:].to_csv(path + '../../submission/top3_most_similar_paper_mixed/valid.cos.{}.{}.mix.{}.{}.{}.top{}.csv'.format(info1,feature_type1,info2, feature_type2, batch_size,top_k),columns=['description_id', 'top1', 'top2', 'top3'], index=False, header=None)

if __name__ == '__main__':
    # old_top3_most_similar_paper()
    # top3_most_similar_paper()
    # top3_most_similar_paper_excluding_training_papers()
    # top3_most_similar_paper_mixed()
    pass

