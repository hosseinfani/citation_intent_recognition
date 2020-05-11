import pandas as pd
import numpy as np
import logging, sys, multiprocessing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('top_k_random_mix')

path = '../output/topk/random/'
path = 'D:/output/topk/random/'
feature_types = ['bm.1_1_gram', 'bm.1_2_gram', 'tfidf.1_1_gram', 'tfidf.1_2_gram']
paper_info = ['title', 'abstract']
k = 100

for info1 in paper_info:
    for feature_type1 in feature_types:
        for info2 in paper_info:
            for feature_type2 in feature_types:
                if info1 == info2 and feature_type1 == feature_type2:
                    continue
                logger.info('train.cos.{}.{}.1000.top{} mixing with train.cos.{}.{}.1000.top{} ...'.format(info1, feature_type1, k, info2, feature_type2, k))
                topk_a = pd.read_csv(path + 'train.cos.{}.{}.1000.top{}'.format(info1, feature_type1, k))
                topk_b = pd.read_csv(path + 'train.cos.{}.{}.1000.top{}'.format(info2, feature_type2, k))

                for (idx_a, a),(idx_b, b) in zip(topk_a.iterrows(), topk_b.iterrows()):
                    merged_scores = [float(a) for a in a.score.split(',')] + [float(b) for b in b.score.split(',')]
                    merged_paper_ids = a.topk.split(',') + b.topk.split(',')
                    ind = np.argsort(merged_scores)[:-((2*k) + 1): -1]
                    scores = np.asarray(merged_scores)[[ind]]
                    paper_ids = np.asarray(merged_paper_ids)[[ind]]

                    paper_ids_unique = []
                    scores_unique = []
                    for i, p in enumerate(paper_ids):
                        if p not in paper_ids_unique:
                            paper_ids_unique.append(p)
                            scores_unique.append(scores[i])

                    topk_a.loc[idx_a]['topk'] = ','.join(paper_ids_unique[:k])
                    topk_a.loc[idx_b]['score'] = ','.join(str(s) for s in scores_unique[:k])

                topk_a.to_csv(path + 'train.cos.{}.{}.mix.{}.{}.1000.top{}'.format(info1, feature_type1, info2, feature_type2, k), columns=['paper_id', 'score', 'topk'], index=False)


