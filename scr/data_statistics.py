"""
paper header & sample
abstract,journal,keywords,paper_id,title,year
NO_CONTENT,Journal of economic entomology,,55a38b7f2401aa93797cef61,Anopheles stephensi...,1978
"We describe...", Infection,, 55a5c23a612c6b12ab2d16e0, Helicobacter cinaedi ..., 2004

description header & sample:
description_id,paper_id,description_text
77bef2,5c0f7919da562944ac759a0f,"Angiogenesis...[[**##**]] ...."

"""

import pandas as pd

df_train = pd.read_csv("../../../dataset/train.csv")
df_valid = pd.read_csv("../../../dataset/validation.csv")
df_data = pd.concat((df_train,df_valid))

df_paper = pd.read_csv("../../../dataset/candidate_paper_for_wsdm2020.csv")

def get_no_abstract_count(df_paper):
    n_noabstract = len(df_paper[df_paper.abstract == 'NO_CONTENT'])
    return n_noabstract, len(df_paper) - n_noabstract

def get_abstract_features(abstracts):
    pass

def get_empty_venue_count(df_paper):
    return len(df_paper[df_paper.keywords == "no-content" or df_paper.keywords.isnull()])

def get_paper_venue_hist(df_paper):
    # % matplotlib
    hist = df_paper.journal.hist(bins=100)#may not make sense

def get_paper_citation_count(df_paper, df_train):
    n_citation = df_train.groupby(['paper_id']).agg(['count'])
    return df_train.set_index('paper_id').join(n_citation)


def get_empty_keywords_count(df_paper):
    return len(df_paper[df_paper.keywords == ""])

def get_paper_publication_year_hist(df_paper):
    # % matplotlib
    hist = df_paper.year.hist(bins=100)

def get_paper_id_hist(df_train):
    return df_data.paper_id.hist(bins=100)

def get_ref_count_hist(df_data):
    hist = df_data.description_text.str.count('[[**##**]]').hist(bins=10)

