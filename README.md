# NeuralSum
Abstractive Sentence Summarization with Attentive Neural Techniques. Presents a new way to evaluate auto-generated abstractive summaries. If you want to use this library to generate sentence summaries, contact us by email for the required data folder. Otherwise, you have to provide your own data, a way to process it, and change the configurations to point to where this data is located.  

## Using Tensor2Tensor  
This project uses the Tensor2Tensor library. The primary run script, `t2t.sh`, can be modified to utilize different models, data, and other parameters. For model-specific parameters, edit `/NeuralSum/NeuralSum/my_custom_hparams.py`. If you add custom hyperparameters, models, modalities, or problems, make sure they are all registered. For info on how to do that, take a look at the  [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) source code and docs.

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
5. Score the generated summaries against the target summaries and print out a report with the results. If `save_report=True`, saves a json file to the reports folder. This script and all the evaluation metrics will soon be moved to a different repository.
Relevant command: `>>> python evaluate_on_duc.py [--which_duc=(2003|2004)] [--save_report=(True|False)]`

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

##  Evaluation Process  
ROUGE scoring has limitations when used for abstractive summarization. It does not accurately reward paraphrasing or other human-like elements of a summary. A bad paraphrase is incorrectly given the same score as a good paraphrase. To fix this problem, we need an evaluation metric that understands sentence semantics.  

To this end, we introduce Facebook Research's InferSent Natural Language Inference tool to summarization evaluation. This tool is a pretrained model capable of converting sentences to sentence embeddings that contain semantic understanding of the original words. This specific model is the state of the art, performing the highest on NLI benchmarks. Thus, it is the best to generate sentence embeddings with. Once the sentence embeddings have been generated for both the generated summary and the target summary, a similarity score can be calculated using cosine similarity. Cosine similarity ranges from 0 (lowest) to 1 (highest). It solves the dot product for the cosine of the angle between the vectors in the vector space. Here are some examples where this alternative evaluation tool produces better judgment than ROUGE:  

#### Example 1: a simple but acceptable paraphrase.  
The word `segments` was replaced with the reasonably similar word `sections`.

Generated summary:  
`Endeavour astronauts join two sections of International Space Station`  
Target summary:  
`Endeavour astronauts join two segments of International Space Station`  
Evaluation Scores:  

| Metric  | Score |  
| :----:  | :---: |  
| ROUGE-1 | 88.89 |  
| ROUGE-2 | 75.00 |  
| ROUGE-l | 88.89 |  
| Cos-Sim | 0.979 |  
|   WMD   | 0.418 |  

#### Example 2: a simple and incorrect paraphrase.  
The word `join` was replaced with the opposite word `remove`.

Generated summary:  
`Endeavour astronauts remove two segments of International Space Station`  
Target summary:  
`Endeavour astronauts join two segments of International Space Station`  
Evaluation Scores:  

| Metric  | Score |  
| :----:  | :---: |  
| ROUGE-1 | 88.89 |  
| ROUGE-2 | 75.00 |  
| ROUGE-l | 88.89 |  
| Cos-Sim | 0.924 |  
|   WMD   | 0.512 |  

#### Example 3: Identical sentences for completeness.  
Generated & target summary:  
`Endeavour astronauts join two segments of International Space Station`  
Evaluation Scores:  

| Metric  | Score  |  
| :----:  | :---:  |  
| ROUGE-1 | 100.00 |  
| ROUGE-2 | 100.00 |  
| ROUGE-l | 100.00 |  
| Cos-Sim |  0.999 |  
|   WMD   |  0.000 |  

#### Conclusion  
Notice that the ROUGE scores of examples 1 & 2 stayed the exact same, not differentiating between a clearly better summary and a clearly worse summary. On the other hand, the cosine similarity was able to identify the difference the bad summary showed, punishing the similarity score by 7.6%. The acceptable summary in Example 1 was only punished 2.1%. When looking at the Word Mover Distance (WMD), we see that the bad summary was given a 24% larger distance from the target than the acceptable summary (0.512 vs 0.418). Thus WMD also shows the ability to judge semantically. Cos-Sim and WMD metrics are different from each other: Cos-Sim is a neural approach using sentence vectors while WMD is an aggregated distance measurement between a sentence's word vectors. Further distinguishing them is the source of word vectors: GloVe for Cos-Sim, and Word2Vec for WMD. Cos-Sim is of course using cosine similarity while WMD uses Euclidean distance. Finally, Cos-Sim is a value to be maximized whereas WMD is a value to be minimized. Because of these differences, both metrics provide value in analyzing abstractive summaries.  

## Contact  
Author: Jacob Krantz  
Email: jkrantz@zagmail.gonzaga.edu  

## Acknowledgements  

This code has been developed for research purposes at the University of Colorado at Colorado Springs. This is an REU project with work supported by the National Science Foundation under Grant No. 1659788.  
Advisor: Dr. Jugal Kalita   

Facebook Research's [InferSent](https://github.com/facebookresearch/InferSent) codebase.  
Full implementation of ROGUE metric [here](https://github.com/pltrdy/rouge).  
Word Mover Distance implementation by [GenSim](https://radimrehurek.com/gensim/models/keyedvectors.html).  
[GloVe](https://nlp.stanford.edu/projects/glove/) embeddings by Stanford NLP group.  
[Word2Vec](https://code.google.com/archive/p/word2vec/) embeddings from Google.  
DUC2003 and [DUC2004](https://duc.nist.gov/duc2004/) conference data by NIST.  
