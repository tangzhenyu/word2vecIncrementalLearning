# Lexical Relational + CBOW Word Embedding
Try to utilize synonyms corpus antonyms corpus and triple corpus in the CBOW model trainng, for better word embedding, which can discover more complex word relation representation and get better performance in synonyms and antonyms recognition.

## Running
Usage is the same as original wrd2vec, just need to provide data train.txt, enwiki9 is used here.

	$ cd gen_data
	$ ./get_train_data.sh

Additional need:synonyms corpus synonym.txt, antonyms corpus antonym.txt, triple corpus triplet.txt。
	
	$ cd src
	$ make
	$ ./lrcwe -train train.txt -synonym synonym.txt -antonym antonym.txt -triplet triplet.txt -output  -save-vocab vocab.txt -belta-rel 0.8 -alpha-rel 0.001 -belta-syn 0.7 -alpha-syn 0.05 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 1 -cbow 1 -iter 3
