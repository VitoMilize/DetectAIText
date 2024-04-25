import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedTokenizerFast

DATA_PATH = 'data'
PREDICTIONS_PATH = 'data/results.csv'
LOGS_PATH = 'data/log_file.log'
MODEL_PATH = '../../Desktop/model'
TOKENIZER_PATH = 'tokenizer'
TFIDF_TRAIN_DATASET_PATH = '../../Desktop/datasets/tfidf_train_essays.csv'
LLM_TRAIN_DATASET_PATH = '../../Desktop/datasets/llm_train_essays.pkl'


class My_Classifier_Model:
    def __init__(self):

        logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(LOGS_PATH)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        self.logger = logger

    def inference_tfidf(self, path_to_dataset):
        logger = self.logger

        try:
            logger.info('Reading dataset...')

            test_df = pd.read_csv(path_to_dataset)

            logger.info('Dataset has been read.')
        except:
            logger.error('Dataset not found.')
            return

        train_df = pd.read_csv(TFIDF_TRAIN_DATASET_PATH)

        try:
            logger.info('Creating tokenizer...')

            raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            raw_tokenizer.normalizer = normalizers.NFC()
            raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

            VOCAB_SIZE = 30522
            special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
            trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

            raw_tokenizer.train_from_iterator(test_df[['text']].values, trainer=trainer)

            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw_tokenizer,
                unk_token="[UNK]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]",
            )

            logger.info('Tokenizer has been created.')
        except:
            logger.error('Tokenizer not created.')
            return

        try:
            logger.info('Tokenization texts...')

            tokenized_texts_test = []
            for text in tqdm(test_df['text'].tolist()):
                tokenized_texts_test.append(tokenizer.tokenize(text))

            tokenized_texts_train = []
            for text in tqdm(train_df['text'].tolist()):
                tokenized_texts_train.append(tokenizer.tokenize(text))

            logger.info('Texts have been tokenized.')
        except:
            logger.error('Texts not tokenized.')
            return

        try:
            logger.info('Transformation tokens to TF-IDF...')

            vectorizer = TfidfVectorizer(
                ngram_range=(3, 5),
                lowercase=False,
                sublinear_tf=True,
                analyzer='word',
                tokenizer=lambda x: x,
                preprocessor=lambda x: x,
                token_pattern=None,
                strip_accents='unicode'
            )

            vectorizer.fit(tokenized_texts_test)

            vocab = vectorizer.vocabulary_

            vectorizer = TfidfVectorizer(
                ngram_range=(3, 5),
                lowercase=False,
                sublinear_tf=True,
                vocabulary=vocab,
                analyzer='word',
                tokenizer=lambda x: x,
                preprocessor=lambda x: x,
                token_pattern=None,
                strip_accents='unicode'
            )

            tf_train = vectorizer.fit_transform(tokenized_texts_train)
            tf_test = vectorizer.transform(tokenized_texts_test)

            y_train = train_df['generated']

            logger.info('Tokens have been transformed to TF-IDF.')
        except:
            logger.error('Tokens not transformed to TF-IDF.')
            return

        try:
            logger.info('Creating classifier...')

            clf = MultinomialNB(alpha=0.02)
            sgd_model = SGDClassifier(
                max_iter=8000,
                tol=1e-4,
                loss="modified_huber"
            )
            p6 = {'verbose': -1, 'learning_rate': 0.005689066836106983,
                  'colsample_bytree': 0.8915976762048253, 'colsample_bynode': 0.5942203285139224,
                  'lambda_l1': 7.6277555139102864, 'lambda_l2': 6.6591278779517808, 'min_data_in_leaf': 156,
                  'max_depth': 11, 'max_bin': 813}
            lgb = LGBMClassifier(**p6)

            ensemble = VotingClassifier(
                estimators=[('mnb', clf), ('sgd', sgd_model), ('lgb', lgb)],
                weights=[0.3, 0.3, 0.4],
                voting='soft',
                n_jobs=-1
            )

            logger.info('Classifier has been created.')
        except:
            logger.error('Classifier not created.')
            return

        try:
            logger.info('Training classifier...')

            ensemble.fit(tf_train, y_train)

            logger.info('Classifier has been trained.')
        except:
            logger.error('Classifier not trained.')
            return

        try:
            logger.info('Creating predictions...')

            predictions = ensemble.predict_proba(tf_test)[:, 1]

            logger.info('Predictions have been created.')
        except:
            logger.error('Predictions not created.')
            return

        try:
            logger.info('Saving predictions...')

            pd.DataFrame({'id': test_df['id'], 'generated': predictions}).to_csv(PREDICTIONS_PATH, index=False)

            logger.info('Predictions have been saved.')
        except:
            logger.error('Predictions not saved.')
            return

    def inference_llm(self, path_to_dataset):
        MAX_LEN = 1024
        BATCH_SIZE = 16
        logger = self.logger

        try:
            logger.info('Creating model...')

            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH,
                                                                       max_position_embeddings=MAX_LEN)

            logger.info('Model has been created.')
        except:
            logger.error('Model not found.')
            return

        try:
            logger.info('Reading dataset...')

            test_df = pd.read_csv(path_to_dataset)

            logger.info('Dataset has been read.')
        except:
            logger.error('Dataset not found.')
            return

        X_test = test_df['text']

        try:
            logger.info('Creating predictions...')

            y_pred = []
            with torch.no_grad():
                for i in tqdm(range(0, len(X_test), BATCH_SIZE)):
                    inputs = tokenizer(
                        X_test[i: i + BATCH_SIZE].tolist(),
                        padding=True,
                        truncation=True,
                        max_length=MAX_LEN,
                        return_tensors='pt',
                    )
                    logits = model(**inputs).logits.numpy()
                    y_pred.extend((np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True))[:, 1])

            del tokenizer
            del model

            logger.info('Predictions have been created.')
        except:
            logger.error('Predictions not created.')
            return

        try:
            logger.info('Saving predictions...')

            pd.DataFrame({'id': test_df['id'], 'generated': y_pred}).to_csv('submission.csv', index=False)

            logger.info('Predictions have been saved.')
        except:
            logger.error('Predictions not saved.')
            return


if __name__ == '__main__':
    os.makedirs(DATA_PATH, exist_ok=True)

    model = My_Classifier_Model()

    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--pipeline')
    parser.add_argument('--dataset')

    args = parser.parse_args()

    if args.dataset:
        if args.pipeline == 'llm':
            if args.command == 'predict':
                model.inference_llm(args.dataset)
            else:
                model.logger.error("Command is required.")
        elif args.pipeline == 'tfidf':
            if args.command == 'predict':
                model.inference_tfidf(args.dataset)
            else:
                model.logger.error("Command is required.")
        else:
            model.logger.error("Pipline is required.")
    else:
        model.logger.error("Dataset is required.")
