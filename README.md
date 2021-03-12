# The code and data for the paper "Multi-Instance Multi-Label Learning Networks for Aspect-Category Sentiment Analysis"

# Requirements
- Python 3.6.8
- torch==1.2.0
- pytorch-transformers==1.1.0
- allennlp==0.9.0

# Instructions:
Before excuting the following commands, replace glove.840B.300d.txt(http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip), bert-base-uncased.tar.gz(https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) and vocab.txt(https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt) with the corresponding absolute paths in your computer. 

## AC-MIMLLN

### Rest14 and Rest14-hard
#### Train
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 32 --train True --evaluate False --evaluation_on_instance_level False

#### Evaluate
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 32 --train False --evaluate True --evaluation_on_instance_level False

#### Performance on Instance Level
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 32 --train False --evaluate False --evaluation_on_instance_level True

### MAMSACSA
#### Train
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 64 --train True --evaluate False --evaluation_on_instance_level False

#### Evaluate
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 64 --train False --evaluate True --evaluation_on_instance_level False

#### Performance on Instance Level
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 64 --train False --evaluate False --evaluation_on_instance_level True

## AC-MIMLLN-Affine
### Rest14 and Rest14-hard
#### Train
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer fc --lstm_layer_num_in_lstm 3 --batch_size 32 --train True --evaluate False --evaluation_on_instance_level False

#### Evaluate
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer fc --lstm_layer_num_in_lstm 3 --batch_size 32 --train False --evaluate True --evaluation_on_instance_level False

#### Performance on Instance Level
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer fc --lstm_layer_num_in_lstm 3 --batch_size 32 --train False --evaluate False --evaluation_on_instance_level True

### MAMSACSA
#### Train
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer fc --lstm_layer_num_in_lstm 3 --batch_size 64 --train True --evaluate False --evaluation_on_instance_level False

#### Evaluate
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer fc --lstm_layer_num_in_lstm 3 --batch_size 64 --train False --evaluate True --evaluation_on_instance_level False

#### Performance on Instance Level
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer fc --lstm_layer_num_in_lstm 3 --batch_size 64 --train False --evaluate False --evaluation_on_instance_level True

## AC-MIMLLN-BERT
### Rest14 and Rest14-hard
#### Train
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path vocab.txt --data_type mil-bert --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert True --pair True --joint_type warmup --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 16 --train True --evaluate False --evaluation_on_instance_level False

#### Evaluate
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path vocab.txt --data_type mil-bert --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert True --pair True --joint_type warmup --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 16 --train False --evaluate True --evaluation_on_instance_level False

#### Performance on Instance Level
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path vocab.txt --data_type mil-bert --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert True --pair True --joint_type warmup --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 16 --train False --evaluate False --evaluation_on_instance_level True

### MAMSACSA
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path vocab.txt --data_type mil-bert --current_dataset MAMSACSA --mil True --bert True --pair True --joint_type warmup --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 16 --train True --evaluate False --evaluation_on_instance_level False

#### Evaluate
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path vocab.txt --data_type mil-bert --current_dataset MAMSACSA --mil True --bert True --pair True --joint_type warmup --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 16 --train False --evaluate True --evaluation_on_instance_level False

#### Performance on Instance Level
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path vocab.txt --data_type mil-bert --current_dataset MAMSACSA --mil True --bert True --pair True --joint_type warmup --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 16 --train False --evaluate False --evaluation_on_instance_level True

## Visualization
After models are trained, we can visualize the attention weights and the word sentiment prediction results by adding two extra options to the commands mentioned above, --train False and --visualize_attention True. For example,

python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer fc --lstm_layer_num_in_lstm 3 --batch_size 32 --train False --visualize_attention True

## Run the models for multiple times
In order to run the models for multiple times, we can use the shell script, repeat.sh, to run the commands mentioned above by replacing the "python" in the commands  with:

sh repeat.sh model_name_0,model_name_1

For example, the following commands train five AC-MIMLLN models and evaluate their performances on the Rest14 dataset, respectively.

sh repeat.sh 0-0-0,0-0-1,0-0-2,0-0-3,0-0-4 nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 32 --train True --evaluate False --evaluation_on_instance_level False

sh repeat.sh 0-0-0,0-0-1,0-0-2,0-0-3,0-0-4 nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py  --embedding_filepath glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 32 --train False --evaluate True --evaluation_on_instance_level False

## Citation
```
@inproceedings{li-etal-2020-multi-instance,
    title = "Multi-Instance Multi-Label Learning Networks for Aspect-Category Sentiment Analysis",
    author = "Li, Yuncong  and
      Yin, Cunxiang  and
      Zhong, Sheng-hua  and
      Pan, Xu",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.287",
    doi = "10.18653/v1/2020.emnlp-main.287",
    pages = "3550--3560",
    abstract = "Aspect-category sentiment analysis (ACSA) aims to predict sentiment polarities of sentences with respect to given aspect categories. To detect the sentiment toward a particular aspect category in a sentence, most previous methods first generate an aspect category-specific sentence representation for the aspect category, then predict the sentiment polarity based on the representation. These methods ignore the fact that the sentiment of an aspect category mentioned in a sentence is an aggregation of the sentiments of the words indicating the aspect category in the sentence, which leads to suboptimal performance. In this paper, we propose a Multi-Instance Multi-Label Learning Network for Aspect-Category sentiment analysis (AC-MIMLLN), which treats sentences as bags, words as instances, and the words indicating an aspect category as the key instances of the aspect category. Given a sentence and the aspect categories mentioned in the sentence, AC-MIMLLN first predicts the sentiments of the instances, then finds the key instances for the aspect categories, finally obtains the sentiments of the sentence toward the aspect categories by aggregating the key instance sentiments. Experimental results on three public datasets demonstrate the effectiveness of AC-MIMLLN.",
}
```