# NeuralSum
Abstractive Sentence Summarization with Attentive Neural Techniques. Presents a new way to evaluate auto-generated abstractive summaries. If you want to use this library to generate sentence summaries, contact us by email for the required data folder. Otherwise, you have to provide your own data, a way to process it, and change the configurations to point to where this data is located.  

## Using Tensor2Tensor  
This project uses the Tensor2Tensor library. The primary run script, `t2t.sh`, can be modified to utilize different models, data, and other parameters. For model-specific parameters, edit `/NeuralSum/NeuralSum/my_custom_hparams.py`. If you add custom hyperparameters, models, modalities, or problems, make sure they are all registered. For info on how to do that, take a look at the  [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) source code and docs. I found Tensor2Tensor to be very particular about configurations. I ran all experiments on a single GPU with 12GB VRAM. Changes must be made for other hardware set-ups. 

#### Overview: Sequence of Events  
Follow these steps for end to end model usage. The script `multirun.sh` does each of these steps for each experiment number specified.  
1. Generate the data to be trained and tested with. Make sure the proper T2TProblem is selected.  
Relevant command: `>>> ./t2t.sh datagen`
2. Train the model on the data that has already been generated. This requires providing a T2TProblem, a T2TModel, and a registered HParams set.
Relevant command: `>>> ./t2t.sh train`
3. Prepare the test data to be tested against. This is the parsing of either DUC2003 or DUC2004 from their raw folders to flat files containing the sentences and the target summaries.
Relevant command: `>>> python process_duc.py [--which_duc=(2003|2004)]`
4. Generate summaries from the trained model given the sentences of either DUC2003 or DUC2004. Make sure `DECODE_FILE` and `DECODE_FILE_OUT` in `t2t.sh` point to the DUC folder desired.
Relevant command: `>>> ./t2t.sh decode`
5. Score the generated summaries against the target summaries and print out a report with the results.  
    **If on branch master:**  follow the instructions on the [VertMetric](https://github.com/jacobkrantz/VertMetric) repository. 
    **If on branch with_vert:** This allows for calculating vert scores from within this repo. Deprecated. If `save_report=True`, saves a json file to the reports folder. This script and all the evaluation metrics will soon be moved to a different repository. Relevant command: `>>> python evaluate_on_duc.py [--which_duc=(2003|2004)] [--save_report=(True|False)]`

#### Hyperparameter Sets  
Hyperparameter sets (Hparams) are objects containing all parameters except certain decoding params. All sets are found in `my_custom_hparams.py`.  
- `exp_6`: base parameters for the Transformer. All other hyperparameter sets are based off this set.  
- `exp_n`: subsequent experiments. Currently hparams set `exp_27` generates the highest ROUGE and VERT scores if you play with certain decoding parameters and training step numbers.  

#### T2T Problems  
Problems define the data used and which metrics to be included for EVAL mode.  
- `summary_problem`: uses all but the last 10k pairs of the Gigaword dataset (3793957 articles).
- `summary_problem_small`: uses the first 25% of Gigaword for training (950989 articles).  

## Datasets  
All datasets are stored in a specific way in the ./data folder. You can ask the maintainers for access to this data.  
- DUC2003  
- DUC2004  
- English Gigaword  
- GloVe embeddings  
- Word2Vec embeddings  

## Unit Tests  
Could use some work. To run all unit tests, execute:  
`>>> python -m unittest discover -s tests`  

## Contact  
Author: Jacob Krantz  
Email: jkrantz@zagmail.gonzaga.edu  

## Acknowledgements  

Did this project help you with your research?  
If so, please considering citing our ICON-2018 paper over on [Arxiv](https://arxiv.org/abs/1810.08838). Thank you!  

This code has been developed for research purposes at the University of Colorado at Colorado Springs. This is an REU project with work supported by the National Science Foundation under Grant No. 1659788.  
Advisor: Dr. Jugal Kalita   

DUC2003 and [DUC2004](https://duc.nist.gov/duc2004/) conference data by NIST.  
