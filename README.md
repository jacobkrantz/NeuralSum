# NeuralSumm
Sentence Summarization with Attentive Neural Techniques


## todo  
extract first sentence of DUC2004 articles and store then in the /clean folder.  
Flatten /docs to just have all the files, no folders.  

## Unit Tests  
To run all unit tests, execute:  
`python -m unittest discover -s tests`  

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
Notice that the ROUGE scores of examples 1 & 2 stayed the exact same, not differentiating between a clearly better summary and a clearly worse summary. On the other hand, the cosine similarity was able to identify the difference the bad summary showed, punishing the similarity score by 7.6%. The acceptable summary in Example 1 was only punished 2.1%. When looking at the Word Mover Distance (WMD), we see that the bad summary was given a 24% larger distance from the target than the acceptable summary (0.512 vs 0.418). Thus WMD also shows the ability to judge semantically. Cos-Sim and WMD metrics are different from each other: Cos-Sim is a neural approach using sentence vectors while WMD is an aggregated distance measurement between a sentence's word vectors. Further distinguishing them is the source of word vectors: GloVe for Cos-Sim, and Word2Vec for WMD. Finally, Cos-Sim is a value to be maximized whereas WMD is a value to be minimized. Because of these differences, both metrics provide value in analyzing abstractive summaries.  
