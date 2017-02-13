# word2veconlinelearning
The code is based on the Google word2vec,and add online learning function.
this version only support negative sampling,the hierarchical softmax function will add int the future version

From the newly added function,every word has its own alpha parameter,which update sepearately through the training process.
In some experiments,this different updating strategy are outperforming fastText and original word2vec in word-similarity tasks.
If you need some new feature of this code,please feel free to contact me,my email is:tangzhenyu1990@bupt.edu.cn
Please feel free to try this code in your practical tasks.

#Usage
run*.sh is the scripts for running word2vec.
Everyone can get some running information from these scripts.
