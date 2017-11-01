# Creating and using custom models in allenNLP

This repository contains experiments for defining custom models in allenNLP and integrating them with the command line tools provided by the allenNLP framework.

Currently, this repository includes:
* A simple Sequence classification model (`custom/models.py`) with a configurable Seq2Vec encoder (from the ones provided in `allennlp/modules/seq2vec_encoders` (https://github.com/allenai/allennlp/tree/master/allennlp/modules/seq2vec_encoders))
* A simple jsonl reader (`custom/readers.py`) with configurable input and label fields
* A predictor to be used for the Sequence classifier (`custom/predictors.py`)
* A script setting up the predictor and wrapping a call to main() from the command line module of allenNLP
* A folder with experiment definitions (`experiments/definitions`), including a simple BagOfEmbeddingClassifier for a really small subset of the Amazon Reviews dataset (`data`)

Assuming you have already installed allenNLP following https://github.com/allenai/allennlp, you can run the following

* Train: `python -m custom.run train experiments/definitions/sequence_boe_classifier.json -s experiments/ouput/boe`
* Predict: `python -m custom.run predict experiments/output/boe/model.tar.gz data/amazon_reviews_video_games_5-1000.json.dev`(of course this should be done with a different set of data, not used for training/evaluating the model, this is just for demonstration purposes
