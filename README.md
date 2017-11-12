# word2veconlinelearning
The code is based on the Google word2vec, with online learning function added. This version only support negative sampling, and the hierarchical softmax function will be added in the future version.
From the newly added function,every word has its own learning parameter,which update sepearately through the training process. In some experiments, this different updating strategy goes beyond the fastText and original word2vec in word-similarity tasks. If you need some new modifications of this code,please feel free to contact me at any time,my email is:tangzhenyu1990@bupt.edu.cn Please feel free to try this code in your practical tasks.

# Usage
run*.sh is the scripts for running word2vec.
Everyone can get some running information from these scripts.

# TransE + Word2vec
Adding triple information, like from freebase, when train Word2vec model, which make better word embedding.

# Lexical Relational + Word2vec
Try to utilize synonyms corpus antonyms corpus and triple corpus in the CBOW model trainng, for better word embedding, which can discover more complex word relation representation and get better performance in synonyms and antonyms recognition.
