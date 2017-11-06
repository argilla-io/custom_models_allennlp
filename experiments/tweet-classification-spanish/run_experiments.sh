#!/bin/bash
export PYTHONHASHSEED=2157
python -m custom.run train experiments/tweet-classification-spanish/definitions/baseline_boe_classifier.json -s experiments/tweet-classification-spanish/output/boe_simple &
python -m custom.run train experiments/tweet-classification-spanish/definitions/baseline_boe_classifier_fasttext_embeddings_tunable.json -s experiments/tweet-classification-spanish/output/boe_emb_tunable &
python -m custom.run train experiments/tweet-classification-spanish/definitions/baseline_boe_classifier_fasttext_embeddings_fixed.json -s experiments/tweet-classification-spanish/output/boe_emb_fixed &
python -m custom.run train experiments/tweet-classification-spanish/definitions/cnn_classifier.json -s experiments/tweet-classification-spanish/output/cnn_simple &
python -m custom.run train experiments/tweet-classification-spanish/definitions/cnn_classifier_fasttext_embeddings_tunable.json -s experiments/tweet-classification-spanish/output/cnn_emb_tunable &
python -m custom.run train experiments/tweet-classification-spanish/definitions/cnn_classifier_fasttext_embeddings_fixed.json -s experiments/tweet-classification-spanish/output/cnn_emb_fixed &
python -m custom.run train experiments/tweet-classification-spanish/definitions/gru_classifier.json -s experiments/tweet-classification-spanish/output/gru_simple &
python -m custom.run train experiments/tweet-classification-spanish/definitions/gru_classifier_fasttext_embeddings_tunable.json -s experiments/tweet-classification-spanish/output/gru_emb_tunable &
python -m custom.run train experiments/tweet-classification-spanish/definitions/gru_classifier_fasttext_embeddings_fixed.json -s experiments/tweet-classification-spanish/output/gru_emb_fixed &