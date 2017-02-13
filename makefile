CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -O2  -Wall -funroll-loops -g

all: word2vec word2vec_multiclass word2phrase distance word-analogy compute-accuracy kmeans

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
word2vec_multiclass : word2vec_multiclass.c
	$(CC) word2vec_multiclass.c -o word2vec_multiclass $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
distance_txt : distance_txt.c
	$(CC) distance_txt.c -o distance_txt $(CFLAGS)
kmeans : kmeans.c
	$(CC) kmeans.c -o kmeans $(CFLAGS)
#gcc kmeans.c -o kmeans -lm -pthread -O2  -Wall -funroll-loops

clean:
	rm -rf word_to_vec_new1 word_to_vec_new word2vec word2phrase distance vec_for_wordlist word-analogy compute-accuracy kmeans
