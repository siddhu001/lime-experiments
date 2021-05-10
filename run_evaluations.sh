mkdir -p trustworthiness_outputs
python -u contrastive_data_trusting.py -d multi_polarity_books -a1 logreg -a2 neighbors -k 10 -u .25 -r 2 | tee trustworthiness_outputs/log_books_logreg_neighbors.txt
python -u contrastive_data_trusting.py -d multi_polarity_books -a1 svm -a2 neighbors -k 10 -u .25 -r 2 | tee trustworthiness_outputs/log_books_svm_neighbors.txt
python -u contrastive_data_trusting.py -d multi_polarity_books -a1 tree -a2 neighbors -k 10 -u .25 -r 2 | tee trustworthiness_outputs/log_books_tree_neighbors.txt
python -u contrastive_data_trusting.py -d multi_polarity_books -a1 tree -a2 random_forest -k 10 -u .25 -r 2 | tee trustworthiness_outputs/log_books_tree_rf.txt
python -u contrastive_data_trusting.py -d multi_polarity_books -a1 svm -a2 random_forest -k 10 -u .25 -r 2 | tee trustworthiness_outputs/log_books_svm_rf.txt
python -u contrastive_data_trusting.py -d multi_polarity_books -a1 logreg -a2 random_forest -k 10 -u .25 -r 2 | tee trustworthiness_outputs/log_books_logreg_rf.txt
python -u contrastive_data_trusting.py -d multi_polarity_dvd -a1 logreg -a2 neighbors -k 10 -u .25 -r 2 | tee trustworthiness_outputs/log_dvd_logreg_neighbors.txt
python -u contrastive_data_trusting.py -d multi_polarity_dvd -a1 svm -a2 neighbors -k 10 -u .25 -r 2 | tee trustworthiness_outputs/log_dvd_svm_neighbors.txt
python -u contrastive_data_trusting.py -d multi_polarity_dvd -a1 tree -a2 neighbors -k 10 -u .25 -r 2 | tee trustworthiness_outputs/log_dvd_tree_neighbors.txt
python -u contrastive_data_trusting.py -d multi_polarity_dvd -a1 logreg -a2 random_forest -k 10 -u .25 -r 2 | tee trustworthiness_outputs/log_dvd_logreg_rf.txt
python -u contrastive_data_trusting.py -d multi_polarity_dvd -a1 svm -a2 random_forest -k 10 -u .25 -r 2 | tee trustworthiness_outputs/log_dvd_svm_rf.txt
python -u contrastive_data_trusting.py -d multi_polarity_dvd -a1 tree -a2 random_forest -k 10 -u .25 -r 2 | tee trustworthiness_outputs/log_dvd_tree_rf.txt
