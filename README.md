This repository contains the code to run the experiments present in [this paper](http://arxiv.org/abs/1602.04938). The code here is frozen to what it was when we originally wrote the paper. If you're interested in using LIME, check out [this repository](https://github.com/marcotcr/lime), where we have packaged it up, improved the code quality, added visualizations and other improvements.

Running the commands below should be enough to get all of the results. You need specific versions python, sklearn, numpy, scipy. Install requirements in a virtualenv using:

`pip install -r requirements.txt`

Also, must have `contrastlime` installed

If we forgot something, please email the first author. 

## Experiment in section 5.2:
- DATASET -> 'multi_polarity_books', 'multi_polarity_kitchen', 'multi_polarity_dvd', 'multi_polarity_kitchen' 

  For NewsGroups: '2ng', 'talkng', '3ng'
  
  - '2ng' = Athiest/Christian
  - 'talkng' = Politics - Guns vs Misc
  - '3ng' = windows.misc, ibm.hardware, windows.x
- ALGORITHM -> 'l1logreg', 'tree'
- EXPLAINER -> 'lime', 'parzen', 'greedy' or 'random'
    
        python evaluate_explanations.py --dataset DATASET --algorithm ALGORITHM --explainer EXPLAINER 
### Results (Average Test Recall, out of 1): 
#####20NewsGroups, Athiest/Christian subset: 
 - Logistic Regression: 0.6464
 - Tree: 0.9650

  **Multi Polarity: Books**

| Technique | LR  |  Decision Tree  |
|-----------|-------|--------------- |
| Random    | 18.27 | 18.51 |
| Greedy    | 63.23  | 37.4 | 
| Lime      | 85.12  | 95.15 |


**Multi Polarity: DVDs**

|  | LR     | Decision Tree           |
|-----------|-------|----------|
| Random    | 21.25 | .2067 | 
| Greedy    | 62.88 | 40.46 | 
| Lime    | 90.92 | 96.74 |

## Experiment in section 5.3:
- DATASET -> 'multi_polarity_books', 'multi_polarity_kitchen', 'multi_polarity_dvd', 'multi_polarity_kitchen'
- ALGORITHM -> 'logreg', 'random_forest', 'svm', 'tree' or 'embforest', although you would need to set up word2vec for embforest

        python data_trusting.py -d DATASET -a ALGORITHM -k 10 -u .25 -r NUM_ROUNDS

### Results
All results shown are F1 (in percent form), similar to the LIME paper

**Multi Polarity: Books**

|  | LR     | NN            | RF            | SVM           |
|-----------|-------|---------------|---------------|---------------|
| Random    | 56.9 | x | x | x |
| Greedy    | 14.7  | x | x | x |
| Lime      | 97.0  | x | x | x |

**Multi Polarity: DVDs**

|  | LR     | NN            | RF            | SVM           |
|-----------|-------|---------------|---------------|---------------|
| Random    | x | x | x | x |
| Greedy    | x | x | x | x |
| Lime      | x | x | x | x |

## Experiment in section 5.4:
- NUM_ROUNDS -> Desired number of rounds
- DATASET -> 'multi_polarity_books', 'multi_polarity_kitchen', 'multi_polarity_dvd', 'multi_polarity_kitchen'
- PICK -> 'submodular' or 'random'
Run the following with the desired number of rounds:

        mkdir out_comparing

        python generate_data_for_compare_classifiers.py -d DATASET -o out_comparing/ -k 10 -r NUM_ROUNDS

        python compare_classifiers.py -d DATASET -o out_comparing/ -k 10 -n 10 -p PICK


## Religion dataset:
Available [here](https://github.com/marcotcr/lime-experiments/blob/master/religion_dataset.tar.gz)

## Multi-polarity datasets:
I got them from [here](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/)
