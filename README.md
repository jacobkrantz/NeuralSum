# NeuralSumm
Sentence Summarization with Attentive Neural Techniques


## todo  
extract first sentence of DUC2004 articles and store then in the /clean folder.  
Flatten /docs to just have all the files, no folders.  

## Unit Tests  
To run all unit tests, execute:  
`python -m unittest discover -s tests`  

##  Evaluation Notes  
ROUGE scoring has limitations when used for abstractive summarization. It does not accurately reward paraphrasing or other human-like elements of a summary. A bad paraphrase is incorrectly given the same score as a good paraphrase. To fix this problem, we need an evaluation metric that understands sentence semantics.  

To this end, we introduce Facebook Research's InferSent Natural Language Inference tool to summarization evaluation. This tool is a pretrained model capable of converting sentences to sentence embeddings that contain semantic understanding of the original words. This specific model is the state of the art, performing the highest on NLI benchmarks. Thus, it is the best to generate sentence embeddings with. Once the sentence embeddings have been generated for both the generated summary and the target summary, a similarity score can be calculated using cosine similarity. Cosine similarity solves the dot product for the cosine of the angle between the vectors in the vector space. Here are some examples where this alternative evaluation tool produces better judgement than ROUGE:  

##### Example 1: a simple but acceptable paraphrase.  
The word `segments` was replaced with the reasonably similar word `sections`.

Generated summary:  
`Endeavour astronauts join two sections of International Space Station`  
Target summary:  
`Endeavour astronauts join two segments of International Space Station`  
Evaluation Scores:    
| Metric | Score |  
|---------|-------|  
| ROUGE-1 | 88.89 |  
| ROUGE-2 | 75.00 |  
| ROUGE-l | 88.89 |  
| Cos-Sim | 0.979 |  

##### Example 2: a simple and incorrect paraphrase.  
The word `join` was replaced with the opposite word `remove`.

Generated summary:  
`Endeavour astronauts remove two segments of International Space Station`  
Target summary:  
`Endeavour astronauts join two segments of International Space Station`  
Evaluation Scores:    
| Metric  | Score |  
|   ---   |  ---  |  
| ROUGE-1 | 88.89 |  
| ROUGE-2 | 75.00 |  
| ROUGE-l | 88.89 |  
| Cos-Sim | 0.924 |  

##### Conclusion  
Notice that the ROUGE scores stayed the exact same, not differentiating between a clearly better summary and a clearly worse summary. On the other hand, the cosine similarity was able to identify the difference in the bad summary, punishing the similarity score by 7.6%. The acceptable paraphrase in Example 1 was only punished 2.1%.  
