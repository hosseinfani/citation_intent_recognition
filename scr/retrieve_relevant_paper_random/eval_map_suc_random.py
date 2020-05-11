import pandas as pd
import numpy as np
import xlsxwriter

import sys
sys.path.extend(['./evaluation'])
from eval import mean_ap_k, mean_suc_k

def one_fold():
    path = 'D:/output/'
    # path = '/mnt/sata_disk/hossein/wsdmcup2020/output/'
    k_list = [3, 5, 10, 100]
    random_sample_size = 1000
    top_relevant_paper_size = 100

    feature_types = ['bm.1_1_gram', 'bm.1_2_gram']
    # feature_types += ['tc.1_1_gram', 'tc.1_2_gram'] #it's commented for the vector is not normalized!
    feature_types += ['tf.1_1_gram', 'tf.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    feature_types += ['emd.50', 'emd.100', 'emd.200', 'emd.300']
    feature_types += ['glv.emd.50', 'glv.emd.100', 'glv.emd.200', 'glv.emd.300']
    feature_types += ['d2v.50', 'd2v.100', 'd2v.200', 'd2v.300']

    paper_info = ['title', 'abstract', 'journal']

    workbook = xlsxwriter.Workbook(path + 'topk/random/eval_one_fold.{}.{}.xlsx'.format(random_sample_size, top_relevant_paper_size))
    ws_avg_map_k = workbook.add_worksheet('avg_map_k')
    ws_var_map_k = workbook.add_worksheet('var_map_k')
    ws_avg_suc_k = workbook.add_worksheet('avg_suc_k')
    ws_var_suc_k = workbook.add_worksheet('var_suc_k')
    row = 0
    for info in paper_info:
        for feature_type in feature_types:
            row += 1
            for i, k in enumerate(k_list):
                ws_avg_map_k.write(0, i + 1, 'avg_map@{}'.format(k))
                ws_avg_map_k.write(row, 0, '{}.{}'.format(info, feature_type))
                ws_var_map_k.write(0, i + 1, 'var_map@{}'.format(k))
                ws_var_map_k.write(row, 0, '{}.{}'.format(info, feature_type))
                ws_avg_suc_k.write(0, i + 1, 'avg_suc@{}'.format(k))
                ws_avg_suc_k.write(row, 0, '{}.{}'.format(info, feature_type))
                ws_var_suc_k.write(0, i + 1, 'avg_suc@{}'.format(k))
                ws_var_suc_k.write(row, 0, '{}.{}'.format(info, feature_type))

                predictions = pd.read_csv(path + 'topk/random/train.cos.{}.{}.{}.top{}'.format(info, feature_type, random_sample_size,top_relevant_paper_size), header=0)
                trues = [[t] for t in predictions['paper_id'].tolist()]
                preds = [str(p).split(',') for p in predictions['topk']]

                mean_suc_k(true=trues, pred=preds, k=k)

                print('map@{}:{}.{}:{}'.format(k, info, feature_type, mean_ap_k(true=trues, pred=preds, k=k)))
                print('suc@{}:{}.{}:{}'.format(k, info, feature_type, mean_suc_k(true=trues, pred=preds, k=k)))
                ws_avg_map_k.write(row, i + 1, mean_ap_k(true=trues, pred=preds, k=k))
                ws_avg_suc_k.write(row, i + 1, mean_suc_k(true=trues, pred=preds, k=k))

    workbook.close()

def ten_fold():
    path = 'D:/output/'
    # path = '/mnt/sata_disk/hossein/wsdmcup2020/output/'
    k_list = [3, 5, 10, 100]
    random_sample_size = 1000
    top_relevant_paper_size = 100

    feature_types = ['bm.1_1_gram', 'bm.1_2_gram']
    # feature_types += ['tc.1_1_gram', 'tc.1_2_gram'] #it's commented for the vector is not normalized!
    feature_types += ['tf.1_1_gram', 'tf.1_2_gram']
    feature_types += ['tfidf.1_1_gram', 'tfidf.1_2_gram']
    feature_types += ['emd.50', 'emd.100', 'emd.200', 'emd.300']
    feature_types += ['glv.emd.50', 'glv.emd.100', 'glv.emd.200', 'glv.emd.300']
    feature_types += ['d2v.50', 'd2v.100', 'd2v.200', 'd2v.300']

    paper_info = ['title', 'abstract', 'journal']

    workbook = xlsxwriter.Workbook(path + 'topk/random/eval_ten_fold.{}.{}.xlsx'.format(random_sample_size, top_relevant_paper_size))
    ws_avg_map_k = workbook.add_worksheet('avg_map_k')
    ws_var_map_k = workbook.add_worksheet('var_map_k')
    ws_avg_suc_k = workbook.add_worksheet('avg_suc_k')
    ws_var_suc_k = workbook.add_worksheet('var_suc_k')
    row = 0
    for info in paper_info:
        for feature_type in feature_types:
            row += 1
            for i, k in enumerate(k_list):
                ws_avg_map_k.write(0, i + 1, 'avg_map@{}'.format(k))
                ws_avg_map_k.write(row, 0, '{}.{}'.format(info, feature_type))
                ws_var_map_k.write(0, i + 1, 'var_map@{}'.format(k))
                ws_var_map_k.write(row, 0, '{}.{}'.format(info, feature_type))
                ws_avg_suc_k.write(0, i + 1, 'avg_suc@{}'.format(k))
                ws_avg_suc_k.write(row, 0, '{}.{}'.format(info, feature_type))
                ws_var_suc_k.write(0, i + 1, 'avg_suc@{}'.format(k))
                ws_var_suc_k.write(row, 0, '{}.{}'.format(info, feature_type))
                maps = []
                sucs = []
                for j in range(1, 10):
                    predictions = pd.read_csv(path + 'topk/random/train.cos.{}.{}.{}.top{}.{}'.format(info, feature_type, random_sample_size,top_relevant_paper_size, j), header=0)
                    trues = [[t] for t in predictions['paper_id'].tolist()]
                    preds = [str(p).split(',') for p in predictions['topk']]
                    maps.append(mean_ap_k(true=trues, pred=preds, k=k))
                    sucs.append(mean_suc_k(true=trues, pred=preds, k=k))

                print('map@{}:{}.{}:{}+-{}'.format(k, info, feature_type, np.asarray(maps).mean(),np.asarray(maps).var()))
                print('suc@{}:{}.{}:{}+-{}'.format(k, info, feature_type, np.asarray(sucs).mean(),np.asarray(sucs).var()))
                ws_avg_map_k.write(row, i + 1, np.asarray(maps).mean())
                ws_var_map_k.write(row, i + 1, np.asarray(maps).var())
                ws_avg_suc_k.write(row, i + 1, np.asarray(sucs).mean())
                ws_var_suc_k.write(row, i + 1, np.asarray(sucs).var())

    workbook.close()
if __name__ == '__main__':
    one_fold()
    ten_fold()
