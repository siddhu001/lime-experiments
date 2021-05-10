import sys
import copy
import os
from contrastlime.lime_text import LimeTextExplainer
import numpy as np
import scipy as sp
import json
import random
import sklearn
from sklearn import ensemble
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn.linear_model import Ridge, TweedieRegressor, BayesianRidge
from sklearn.pipeline import make_pipeline
import pickle
import explainers
import embedding_forest
from load_datasets import *
import argparse
import collections
    
def get_classifier(name, vectorizer):
  if name == 'logreg':
    return linear_model.LogisticRegression(fit_intercept=True)
  if name == 'random_forest':
    return ensemble.RandomForestClassifier(n_estimators=1000, random_state=1, max_depth=5, n_jobs=10)
  if name == 'svm':
    return svm.SVC(probability=True, kernel='rbf', C=10,gamma=0.001)
  if name == 'tree':
    return tree.DecisionTreeClassifier(random_state=1)
  if name == 'neighbors':
    return neighbors.KNeighborsClassifier()
  if name == 'embforest':
    return embedding_forest.EmbeddingForest(vectorizer)
  # Should not reach.
  assert False, breakpoint()

def main():
  parser = argparse.ArgumentParser(description='Evaluate some explanations')
  parser.add_argument('--dataset', '-d', type=str, required=True,help='dataset name')
  parser.add_argument('--algorithm1', '-a1', type=str, required=True, help='algorithm_name')
  parser.add_argument('--algorithm2', '-a2', type=str, required=True, help='algorithm_name')
  parser.add_argument('--num_features', '-k', type=int, required=True, help='num features')
  parser.add_argument('--percent_untrustworthy',  '-u', type=float, required=True, help='percentage of untrustworthy features. like 0.1')
  parser.add_argument('--num_rounds', '-r', type=int, required=True, help='num rounds')
  args = parser.parse_args()
  dataset = args.dataset
  train_data, train_labels, test_data, test_labels, class_names = LoadDataset(dataset)

  vectorizer = CountVectorizer(lowercase=False, binary=True) 
  train_vectors = vectorizer.fit_transform(train_data)
  test_vectors = vectorizer.transform(test_data)
  terms = np.array(list(vectorizer.vocabulary_.keys()))
  indices = np.array(list(vectorizer.vocabulary_.values()))
  inverse_vocabulary = terms[np.argsort(indices)]

  np.random.seed(1)
  classifier_a = get_classifier(args.algorithm1, vectorizer)
  classifier_a.fit(train_vectors, train_labels)
  classifier_a_pipeline = make_pipeline(vectorizer, classifier_a)

  classifier_b = get_classifier(args.algorithm2, vectorizer)
  classifier_b.fit(train_vectors, train_labels)
  classifier_b_pipeline = make_pipeline(vectorizer, classifier_b)

  np.random.seed(1)
  untrustworthy_rounds = []
  all_features = range(train_vectors.shape[1])
  num_untrustworthy = int(train_vectors.shape[1] * args.percent_untrustworthy)
  for _ in range(args.num_rounds):
    untrustworthy_rounds.append(np.random.choice(all_features, num_untrustworthy, replace=False))
  
  rho = 25
  kernel = lambda d: np.sqrt(np.exp(-(d**2) / rho ** 2))

  # simple_LIME = explainers.GeneralizedLocalExplainer(kernel, explainers.data_labels_distances_mapping_text, num_samples=15000, return_mean=True, verbose=False, return_mapped=True)
  LIME = LimeTextExplainer(class_names=class_names, mode="classification") 

  ridge_regressor = Ridge(alpha=1, fit_intercept=True, random_state=0)
  model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=0)
  regressor_requires_positive_values=False

  sigmas = {'multi_polarity_electronics': {'neighbors': 0.75, 'svm': 10.0, 'tree': 0.5,
  'logreg': 0.5, 'random_forest': 0.5, 'embforest': 0.75},
  'multi_polarity_kitchen': {'neighbors': 1.0, 'svm': 6.0, 'tree': 0.75,
  'logreg': 0.25, 'random_forest': 6.0, 'embforest': 1.0},
  'multi_polarity_dvd': {'neighbors': 0.5, 'svm': 0.75, 'tree': 8.0, 'logreg':
  0.75, 'random_forest': 0.5, 'embforest': 5.0}, 'multi_polarity_books':
  {'neighbors': 0.5, 'svm': 7.0, 'tree': 2.0, 'logreg': 1.0, 'random_forest':
  1.0, 'embforest': 3.0}, '2ng': {'neighbors': 1.0, 'svm': 6.0, 'tree': 0.75,
  'logreg': 0.25, 'random_forest': 6.0, 'embforest': 1.0}}

  random = explainers.RandomExplainer()
  exps = {}
  explainer_names = ['DiffLIME', 'ContrastLIME', 'random', 'greedy']
  for expl in explainer_names:
    exps[expl] = []

  predictions_a = classifier_a.predict(test_vectors)
  predict_probas_a = classifier_a.predict_proba(test_vectors)[:,1]
  predictions_b = classifier_b.predict(test_vectors)

  disagreements = np.array(predictions_a != predictions_b, dtype=int)

  predict_probas_b = classifier_b.predict_proba(test_vectors)[:,1]

  LARGE_NUM_OF_FEATURES=200

  for i in range(test_vectors.shape[0]):
    print(i)
    sys.stdout.flush()

    # Doesn't need to change between single-model and contrastive LIME.
    exp = random.explain_instance(test_vectors[i], 1, None, args.num_features, None)
    exps['random'].append(exp)

    # Compute Diff-LIME
    class_exp_a = LIME.explain_instance(test_data[i],
                                      classifier_a_pipeline.predict_proba,
                                      num_features=LARGE_NUM_OF_FEATURES,
                                      model_regressor=ridge_regressor)
    lime_exp_a = [(vectorizer.vocabulary_.get(w, None), weight) for w, weight in class_exp_a.as_list() if w in vectorizer.vocabulary_]
    lime_exp_a_dict = dict(lime_exp_a)
    lime_keys_a = set(lime_exp_a_dict.keys())
    lime_score_a = class_exp_a.score
    class_exp_b = LIME.explain_instance(test_data[i],
                                      classifier_b_pipeline.predict_proba,
                                      num_features=LARGE_NUM_OF_FEATURES,
                                      model_regressor=ridge_regressor)
    lime_exp_b = [(vectorizer.vocabulary_.get(w, None), weight) for w, weight in class_exp_b.as_list() if w in vectorizer.vocabulary_]
    lime_exp_b_dict = dict(lime_exp_b)
    lime_keys_b = set(lime_exp_b_dict.keys())
    lime_score_b = class_exp_b.score
    combined_lime_keys = lime_keys_a.union(lime_keys_b)
    diff_lime_exp = []
    for word_idx in combined_lime_keys:
      lime_difference = lime_exp_b_dict.get(word_idx, 0.0) - lime_exp_a_dict.get(word_idx, 0.0)
      diff_lime_exp.append((word_idx, lime_difference))
    # Sort by difference of LIMEs
    diff_lime_exp.sort(key = lambda x: np.abs(x[1]), reverse=True)
    diff_lime_exp = diff_lime_exp[:args.num_features]
    assert lime_score_a.keys() == lime_score_b.keys()
    diff_lime_score = {k: lime_score_b[k] - lime_score_a[k] for k in lime_score_a}
    exps['DiffLIME'].append((diff_lime_exp, diff_lime_score))
    
    # Compute ContrastLime

    contrastlime_class_exp = LIME.explain_instance_contrast(test_data[i],
                                      classifier_a_pipeline.predict_proba,
                                      classifier_b_pipeline.predict_proba,
                                      num_features=args.num_features,
                                      model_regressor=model_regressor,
                                      regressor_requires_positive_values=regressor_requires_positive_values)
    contrastlime_exp = [(vectorizer.vocabulary_.get(w, None), weight) for w, weight in contrastlime_class_exp.as_list() if w in vectorizer.vocabulary_]
    contrastlime_score = contrastlime_class_exp.score
    exps['ContrastLIME'].append((contrastlime_exp, contrastlime_score))

    exp = explainers.explain_contrast_greedy_martens(test_vectors[i],
                                                     disagreements[i],
                                                     classifier_a.predict_proba,
                                                     classifier_b.predict_proba,
                                                     args.num_features)
    exps['greedy'].append(exp)

  precision = {}
  recall = {}
  f1 = {}

  neg_precision = {}
  neg_recall = {}
  neg_f1 = {}

  macro_precision = {}
  macro_recall = {}
  macro_f1 = {}

  for name in explainer_names:
    precision[name] = []
    recall[name] = []
    f1[name] = []
    neg_precision[name] = []
    neg_recall[name] = []
    neg_f1[name] = []
    macro_precision[name] = []
    macro_recall[name] = []
    macro_f1[name] = []
  flipped_preds_size = []
  for untrustworthy in untrustworthy_rounds:
    t = test_vectors.copy()
    t[:, untrustworthy] = 0

    disagreement_predictions_originals = classifier_a.predict(test_vectors) != classifier_b.predict(test_vectors)
    disagreement_predictions_updated = classifier_a.predict(t) != classifier_b.predict(t)
    mistrust_idx = np.argwhere(disagreement_predictions_originals != disagreement_predictions_updated).flatten()

    print('Number of suspect predictions ', len(mistrust_idx))
    shouldnt_trust = set(mistrust_idx)
    flipped_preds_size.append(len(shouldnt_trust))
    mistrust = collections.defaultdict(lambda:set())
    trust = collections.defaultdict(lambda: set())
    trust_fn = lambda prev, curr: (prev > 0.5 and curr > 0.5) or (prev <= 0.5 and curr <= 0.5)
    trust_fn_all = lambda exp, unt: len([x[0] for x in exp if x[0] in unt]) == 0
    for i in range(test_vectors.shape[0]):
      prev_tot = predict_probas_b[i] - predict_probas_a[i]

      exp, mean = exps['DiffLIME'][i]
      assert list(mean.keys()) == [1]
      prev_tot2 = sum([np.abs(x[1]) for x in exp]) + np.abs(mean[1])
      tot = prev_tot2 - sum([np.abs(x[1]) for x in exp if x[0] in untrustworthy])
      trust['DiffLIME'].add(i) if trust_fn(tot, prev_tot) else mistrust['DiffLIME'].add(i)



      exp, mean = exps['ContrastLIME'][i]
      assert list(mean.keys()) == [1]
      prev_tot2 = sum([np.abs(x[1]) for x in exp]) + np.abs(mean[1])
      tot = prev_tot2 - sum([np.abs(x[1]) for x in exp if x[0] in untrustworthy])
      trust['ContrastLIME'].add(i) if trust_fn(tot, prev_tot) else mistrust['ContrastLIME'].add(i)


      exp = exps['random'][i]
      trust['random'].add(i) if trust_fn_all(exp, untrustworthy) else mistrust['random'].add(i)

      exp = exps['greedy'][i]
      trust['greedy'].add(i) if trust_fn_all(exp, untrustworthy) else mistrust['greedy'].add(i)

    for expl in explainer_names:
      # switching the definition
      false_positives = set(trust[expl]).intersection(shouldnt_trust)
      true_positives = set(trust[expl]).difference(shouldnt_trust)
      false_negatives = set(mistrust[expl]).difference(shouldnt_trust)
      true_negatives = set(mistrust[expl]).intersection(shouldnt_trust)

      try:
        prec= len(true_positives) / float(len(true_positives) + len(false_positives))
      except:
        prec= 0
      try:
        rec= float(len(true_positives)) / (len(true_positives) + len(false_negatives))
      except:
        rec= 0
      try:
        neg_prec= len(true_negatives) / float(len(true_negatives) + len(false_negatives))
      except:
        neg_prec= 0
      try:
        neg_rec= float(len(true_negatives)) / (len(true_negatives) + len(false_positives))
      except:
        neg_rec= 0
      precision[expl].append(prec)
      recall[expl].append(rec)
      f1z = 2 * (prec * rec) / (prec + rec) if (prec and rec) else 0
      f1[expl].append(f1z)

      neg_precision[expl].append(neg_prec)
      neg_recall[expl].append(neg_rec)
      neg_f1z = 2 * (neg_prec * neg_rec) / (neg_prec + neg_rec) if (neg_prec and neg_rec) else 0
      neg_f1[expl].append(neg_f1z)

      macro_precision[expl].append((prec + prec)/2)
      macro_recall[expl].append((rec + rec)/2)
      macro_f1[expl].append((f1z + f1z) / 2)



  print('Average number of flipped predictions:', np.mean(flipped_preds_size), '+-', np.std(flipped_preds_size))
  print('Macro Precision:')
  for expl in explainer_names:
    print(expl, np.mean(macro_precision[expl]), '+-', np.std(macro_precision[expl]), 'pvalue', sp.stats.ttest_ind(macro_precision[expl], macro_precision['ContrastLIME'])[1].round(4))
  print()
  print('Macro Recall:')
  for expl in explainer_names:
    print(expl, np.mean(macro_recall[expl]), '+-', np.std(macro_recall[expl]), 'pvalue', sp.stats.ttest_ind(macro_recall[expl], macro_recall['ContrastLIME'])[1].round(4))
  print()
  print('Macro F1:')
  for expl in explainer_names:
    print(expl, np.mean(macro_f1[expl]), '+-', np.std(macro_f1[expl]), 'pvalue', sp.stats.ttest_ind(macro_f1[expl], macro_f1['ContrastLIME'])[1].round(4))

if __name__ == "__main__":
    main()
