# word2veconlinelearning
The code is based on the Google word2vec,and add online learning function.
this version only support negative sampling,the hierarchical softmax function will add int the future version
if you need some new feature of this code,please feel free to contact me,my email is:tangzhenyu1990@bupt.edu.cn

#Usage
1.first time to training the model： 
./word2vec -train train_sample -output vectors.bin11 -cbow 0 -size 200 -window 3 -hs 0  -threads 1 -binary 1 -negative 2 -min-count 2 -model-output model.out11  -save-vocab vocab11.txt -update 0 

2.iterative refinement of preceding model： 
./word2vec -train train_sample -read-vocab vocab11.txt -output vectors.bin22 -cbow 0 -size 200 -window 3 -hs 0 -threads 1 -binary 1 -negative 2 -min-count 2 -model-input model.out11 -model-output model.out22   -save-vocab vocab22.txt -update 1  
