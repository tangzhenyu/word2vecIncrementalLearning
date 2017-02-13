//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

/*
 *Word alpha ---ok
 *Word alpha + global alpha  ---ok
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include <assert.h>
#define MAX_STRING 1000
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

/**
 * cn:
 * word:
 * code:
 * codelen:
 */
struct vocab_word {
	long long cn;
	long long actual_read;
	long long id_old;
	real alpha;
	int *point;
	char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING],
	 model_output_file[MAX_STRING], model_input_file[MAX_STRING],output_file_other[MAX_STRING],vec_input_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab,*vocab_new;
int save_binary = 0,read_binary = 0,cbow = 0, debug_mode = 2, window = 5, min_count = 5,
	num_threads = 1, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 600,
	 vocab_size_old = 0;
long long train_words = 0,train_words_old=0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1, *syn1neg, *expTable, *syn1_old,*syn1neg_old, *syn0_old;
clock_t start;

int hs = 1, negative = 0, update = 0;
const int table_size = 1e8;
//int table_size = 1e8;
const long long max_w = 50; 
int *table;
long long *words_count;
int iterations=1;
void InitUnigramTable() {
	int a, i;
	long long train_words_pow = 0;
	real d1, power = 0.75;
	//printf("%lld %d\n",train_words_pow,vocab[0].cn);
	table = (int *) malloc(table_size * sizeof(int));
	if (table == NULL) 
	{
		printf("Memory allocation failed\n"); 
		exit(1);
	}
	for (a = 0; a < vocab_size; a++){
		train_words_pow += pow(vocab[a].cn, power);
		//printf("%lld\n",train_words_pow);
	}
	i = 0;
	d1 = pow(vocab[i].cn, power) / (real) train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (real) table_size > d1) {
			i++;//指向下一个word
			d1 += pow(vocab[i].cn, power) / (real) train_words_pow;
		}
		if (i >= vocab_size)
			i = vocab_size - 1;
	}
}
/*
   void InitUnigramTable_test() {
   int a, i;
   long long train_words_pow = 0;
   real d1, power = 0.75;

//modification
table_size=vocab_size;

table = (int *) malloc(table_size * sizeof(int));

for (a = 0; a < table_size; a++) {
table[a] = a;
}
}
*/

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13)
			continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if (ch == '\n')
					ungetc(ch, fin);
				break;
			}
			if (ch == '\n') {
				strcpy(word, (char *) "</s>");
				return;
			} else
				continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1)
			a--;   // Truncate too long words
	}
	word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++)
		hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
	unsigned int hash = GetWordHash(word);
	while (1) {
		if (vocab_hash[hash] == -1)
			return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word))
			return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	//printf("%s ",word);
	if (feof(fin))
		return -1;
	return SearchVocab(word);
}

//int count1=0;

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING)
		length = MAX_STRING;
	vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	//	vocab[vocab_size].cn = 1;
	//	vocab[vocab_size].actual_read=0;
	//	vocab[vocab_size].alpha=alpha;

	//++count1;

	//printf("%lld %lld %f\n",vocab[vocab_size].cn,vocab[vocab_size].actual_read,vocab[vocab_size].alpha);
	//vocab[vocab_size].id_old = vocab_size;   //瀹氫箟璇ord鍦ㄤ箣鍓峷ocab閲岀殑涓嬫爣

	//printf("%s %lld\n",vocab[vocab_size].word,vocab[vocab_size].id_old);

	vocab_size++;
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		//vocab = (struct vocab_word *) realloc(vocab,
		//		vocab_max_size * sizeof(struct vocab_word));
		vocab_new = (struct vocab_word *) realloc(vocab,
				vocab_max_size * sizeof(struct vocab_word));
		if(!vocab_new){
			printf("Realloc vocab mem failed!!\n");
			exit(1);
		}
		vocab=vocab_new;

	}
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1)
		hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size - 1;
	return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
	int a, size;
	unsigned int hash;
	// Sort the vocabulary(order by vocab's count desc) and keep </s> at the first position
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);

	/*
	   for(a = 0 ; a < vocab_size ; ++a){
	   printf("%lld %s %lld %lld\n",a,vocab[a].word,vocab[a].cn,vocab[a].actual_read);
	   }
	   exit(-1);
	   */

	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	size = vocab_size;   //
	train_words = 0;
	long k=0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		if (vocab[a].cn < min_count) {
			printf("Reduce vocab\n");
			vocab_size--;
			free(vocab[vocab_size].word);   //free
		} else {
			// Hash will be re-computed, as after the sorting it is not actual

			hash = GetWordHash(vocab[a].word);
			while (vocab_hash[hash] != -1)
				hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = a;   //

			//vocab[a].id_old=k;
			//k++;

			train_words += vocab[a].cn;//count the sum words of prev training file and current file
		}
	}

	train_words -= train_words_old;
	vocab = (struct vocab_word *) realloc(vocab,
			(vocab_size + 1) * sizeof(struct vocab_word));

	if(!vocab){
		printf("Reallocate Failed!!\n");\
			exit(1);
	}

	/*
	   for(a = 0 ; a < vocab_size ; ++a){
	   printf("%lld %s %lld %lld\n",a,vocab[a].word,vocab[a].cn,vocab[a].actual_read);
	   }
	   exit(-1);
	   */
	// Allocate memory for the binary tree construction
	printf("Vocab Size from:%lld to %lld,max size:%lld\n",vocab_size_old,vocab_size,vocab_max_size);
	real factor=0;
	for (a = 0; a < vocab_size; ++a) {
		vocab[a].code = (char *) calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *) calloc(MAX_CODE_LENGTH, sizeof(int));
		factor = 1-vocab[a].actual_read/(real)(vocab[a].cn+1);
		if(factor <= 0){
			vocab[a].alpha=alpha;
		}else{
			vocab[a].alpha=alpha*factor;
		}
		//vocab[a].alpha = alpha * (1-vocab[a].actual_read/(real)(vocab[a].cn+1));
		//printf("%lld %s %lld %lld\n",a,vocab[a].word,vocab[a].cn,vocab[a].actual_read);
		//printf("%lld %s %f %lld %lld\n",a,vocab[a].word,factor,vocab[a].cn,vocab[a].actual_read);
	}
	//exit(-1);
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
	int a, b = 0;
	unsigned int hash;
	for (a = 0; a < vocab_size; a++)
		if (vocab[a].cn > min_reduce) {
			vocab[b].cn = vocab[a].cn;
			vocab[b].word = vocab[a].word;
			vocab[b].alpha = vocab[a].alpha;
			vocab[b].actual_read = vocab[a].actual_read;
			b++;
		} else
			free(vocab[a].word);
	vocab_size = b;
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1)
			hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
	min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
//ocab_size * 2 + 1
//ocab_size * 2 + 1
//count:
void CreateBinaryTree() {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	long long *count = (long long *) calloc(vocab_size * 2 + 1,
			sizeof(long long));
	long long *binary = (long long *) calloc(vocab_size * 2 + 1,
			sizeof(long long));
	long long *parent_node = (long long *) calloc(vocab_size * 2 + 1,
			sizeof(long long));
	//
	for (a = 0; a < vocab_size; a++)
		count[a] = vocab[a].cn;
	for (a = vocab_size; a < vocab_size * 2; a++)
		count[a] = 1e15;
	//
	pos1 = vocab_size - 1;
	pos2 = vocab_size;
	// Following algorithm constructs the Huffman tree by adding one node at a time
	//printf("Test1!!\n");
	for (a = 0; a < vocab_size - 1; a++) {
		// First, find two smallest nodes 'min1, min2'
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			} else {
				min1i = pos2;
				pos2++;
			}
		} else {
			min1i = pos2;
			pos2++;
		}

		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			} else {
				min2i = pos2;
				pos2++;
			}
		} else {
			min2i = pos2;
			pos2++;
		}
		count[vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = vocab_size + a;   //瀹氫箟鐖惰妭鐐�
		parent_node[min2i] = vocab_size + a;   //瀹氫箟鐖惰妭鐐�
		binary[min2i] = 1;
	}
	//printf("Test2!!\n");
	//printf("Vocab Size:%lld!!\n",vocab_size);
	// Now assign binary code to each vocabulary word
	for (a = 0; a < vocab_size; a++) {
		b = a;
		i = 0;
		//code[i]
		//exit(-1);
		//
		while (1) {
			//printf("%lld %lld",binary[b],b);
			code[i] = binary[b];//binary[b] is the code of the bth node,b is the index of the path from leaf node to tree node
			point[i] = b;//b is the index of its parent
			i++;
			b = parent_node[b];
			if (b == vocab_size * 2 - 2)//root node of the binary tree
				break;
		}
		//codelen
		vocab[a].codelen = i;
		//
		vocab[a].point[0] = vocab_size - 2;

		for (b = 0; b < i; b++) {//the code array of the vocab[a] if from tree node to leaf node
			vocab[a].code[i - b - 1] = code[b];//vocab[a].code[0] is root
			vocab[a].point[i - b] = point[b] - vocab_size;
		}
	}
	//	printf("Test3!!\n");
	//	exit(-1);
	free(count);
	free(binary);
	free(parent_node);
}

void ReadVocab_Update() {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin = fopen(read_vocab_file, "rb");
	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	vocab_size = 0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin))
			break;
		a = AddWordToVocab(word);
		fscanf(fin, "%lld\t%lld%c", &vocab[a].cn,&vocab[a].id_old,&c);

		vocab[a].actual_read=vocab[a].cn;
		//vocab[a].alpha=alpha * (1-vocab[a].actual_read/(real)(vocab[a].cn+1));
		train_words_old += vocab[a].cn;
		//printf("%s %lld %lld%c",word,vocab[a].cn,vocab[a].id_old,c);
		//printf("%s ==> %lld\n", vocab[a].word, vocab[a].cn);
		i++;
	}

	//train_words = vocab_size;
	vocab_size_old = vocab_size;
	if (debug_mode > 0) {
		printf("Read Prev Vocab File !!!\n");
		printf("Prev Vocab size: %lld\n", vocab_size_old);
		printf("Words in prev train file: %lld\n", train_words_old);
	}

	//file_size = ftell(fin);
	fclose(fin);
}

void LearnVocabFromTrainFile_Update() {
	char word[MAX_STRING];
	FILE *fin;
	long long a, i;


	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}

	//train_words += vocab_size;
	train_words = 0;
	int new_words=0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin))
			break;
		train_words++;
		if ((debug_mode > 1) && (train_words % 100000 == 0)) {
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		i = SearchVocab(word);

		if (i == -1) {
			a = AddWordToVocab(word);
			//printf("%s----------------------%lld\n",word,a);
			vocab[a].cn = 1;
			vocab[a].id_old=a;
			vocab[a].actual_read=0;
			//vocab[a].alpha=alpha;
			//vocab[a].alpha=alpha * (1-vocab[a].actual_read/(real)(vocab[a].cn+1));
			++new_words;
		} else
			vocab[i].cn++;
		if (vocab_size > vocab_hash_size * 0.7)
			ReduceVocab();
	}



	printf("Vocab Size from:%lld to %lld,max size:%lld\n",vocab_size_old,vocab_size,vocab_max_size);
	printf("New Added words: %d\n",new_words);

	/*
	   for(a = 0 ; a < vocab_size ; ++a){
	   printf("%lld %s %lld %lld\n",a,vocab[a].word,vocab[a].cn,vocab[a].actual_read);
	   }
	   */
	//exit(-1);
	//printf("######################\m");
	SortVocab();
	if (debug_mode > 0) {
		printf("Read Training File !!!\n");
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	file_size = ftell(fin);
	fclose(fin);
}

void LearnVocabFromTrainFile() {
	char word[MAX_STRING];
	FILE *fin;
	long long a, i;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	vocab_size = 0;
	a=AddWordToVocab((char *)"</s>");
	vocab[a].id_old=0;

	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		train_words++;
		if ((debug_mode > 1) && (train_words % 100000 == 0)) {
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		i = SearchVocab(word);
		if (i == -1) {
			//			printf("%s\n",word);
			a = AddWordToVocab(word);
			//printf("%s %lld %lld %lld\n",word,vocab_size,vocab_max_size,a);
			vocab[a].id_old = a;
			//vocab[a].cn = 1;
		} else vocab[i].cn++;
		if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
	}
	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	//	printf("Counts : %d\n",count1);
	file_size = ftell(fin);
	fclose(fin);
}

void SaveVocab() {
	long long i;
	FILE *fo = fopen(save_vocab_file, "wb");
	for (i = 0; i < vocab_size; i++){
		//fprintf(fo, "%s %lld %lld %lld %f\n", vocab[i].word, vocab[i].id_old,vocab[i].cn,
		//		vocab[i].id_old,vocab[i].alpha);
		fprintf(fo, "%s %lld %lld\n", vocab[i].word, vocab[i].cn,
				vocab[i].id_old);
	}
	fclose(fo);
}

void ReadVocab() {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin = fopen(read_vocab_file, "rb");
	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	vocab_size = 0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin))
			break;
		a = AddWordToVocab(word);
		fscanf(fin, "%lld%c", &vocab[a].cn, &c);
		i++;
	}

	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);
	fclose(fin);
}

void InitNet() {
	long long a, b;
	words_count = (long long *) malloc(vocab_size * sizeof(long long));
	for(a = 0 ; a < vocab_size ; ++a){
		words_count[a]=vocab[a].cn;
	}

	a = posix_memalign((void **) &syn0, 128,
			(long long) vocab_size * layer1_size * sizeof(real));

	if (a) {
		printf("Memory allocation failed\n");
		exit(1);
	}

	if (hs) {
		a = posix_memalign((void **) &syn1, 128,
				(long long) vocab_size * layer1_size * sizeof(real));
		if (syn1 == NULL) {
			printf("Memory allocation failed\n");
			exit(1);
		}
		for (b = 0; b < layer1_size; b++)
			for (a = 0; a < vocab_size; a++)
				syn1[a * layer1_size + b] = 0;
	}
	if (negative > 0) {
		a = posix_memalign((void **) &syn1neg, 128,
				(long long) vocab_size * layer1_size * sizeof(real));
		if (syn1neg == NULL) {
			printf("Memory allocation failed\n");
			exit(1);
		}
		for (b = 0; b < layer1_size; b++)
			for (a = 0; a < vocab_size; a++)
				syn1neg[a * layer1_size + b] = 0;
	}

	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
		{
			syn0[a * layer1_size + b] = (rand() / (real) RAND_MAX - 0.5)
				/ layer1_size;
		}

	/*
	   for(a = 0 ; a < vocab_size ; ++a){
	   vocab[a].alpha=(1-vocab[a].actual_read/(real)(vocab[a].cn+1));
	   }
	   */
	//	CreateBinaryTree();
	printf("Complete CreateBinaryTree!!\n");
}


void InitVec(){
	int read_lines = 0;
	int read_counts = 0;
	char ch;
	long long a, b,c;
	long long words, size;
	//	char* vocab_word_point1 = (char *) malloc((long long) max_w * sizeof(char));
	char  vocab_word_point1[max_w];
	//	memset(vocab_word_point1,0,max_w * sizeof(char));
	FILE *fin=fopen(vec_input_file,"rb");

	if (fin == NULL) {
		printf("Model file not found\n");
		exit(1);
	}
	printf("%lld %lld %lld\n",vocab_size_old,vocab_size,layer1_size);	
	//	syn0_old= (real*) malloc((long long) vocab_size_old * layer1_size * sizeof(real));
	//	syn0=(real*) malloc((long long) vocab_size * layer1_size * sizeof(real));

	a = posix_memalign((void **)&syn0_old, 128, (long long)vocab_size_old * layer1_size * sizeof(real));
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
	if(syn0_old == NULL){
		printf("Memory allocation failed\n");
		exit(1);
	}
	if(syn0 == NULL){
		printf("Memory allocation failed\n");
		exit(1);
	}
	fscanf(fin, "%lld", &words);
	fscanf(fin, "%lld", &size);

	printf("%lld %lld\n",words,size);	

	assert(vocab_size_old == words);
	assert(layer1_size == size);
	//	layer1_size=size;
	if (read_binary) {
		for (a = 0; a < vocab_size_old; a++) {
			//fscanf(fin, "%s%c", vocab_word_point, &ch);
			//printf("%s ",vocab_word_point);
			//unsigned long long next_random = 1;
			c=0;
			while (1) {
				vocab_word_point1[c] = fgetc(fin);
				if (feof(fin) || (vocab_word_point1[c] == ' ')) 
					break;
				if ((c < max_w) && (vocab_word_point1[c] != '\n')) 
					c++;
			}             
			vocab_word_point1[c]='\0';
			//string word=string(vocab_word_point1);
			memset(vocab_word_point1,0,max_w);

			for (b = 0; b < layer1_size; b++) {
				fread(&syn0_old[a * layer1_size + b], sizeof(real), 1, fin);
				//printf("%f ",syn0_old[a * layer1_size + b]);
				++read_counts;
			}
			//ch = fgetc(fin);
			//printf("\n");
			++read_lines;
		}
	} else {
		for (a = 0; a < vocab_size_old; a++) {
			//fscanf(fin, "%s%c", vocab_word_point1, &ch);
			c=0;
			while (1) {
				vocab_word_point1[c] = fgetc(fin);
				if (feof(fin) || (vocab_word_point1[c] == ' ')) 
					break;
				if ((c < max_w) && (vocab_word_point1[c] != '\n')) 
					c++;
			}             
			vocab_word_point1[c]='\0';
			//string word=string(vocab_word_point1);
			memset(vocab_word_point1,0,max_w);
			for (b = 0; b < layer1_size; b++) {
				fscanf(fin, "%f", &syn0_old[a * layer1_size + b]);
				read_counts++;
			}
			ch = fgetc(fin);//
			++read_lines;
		}
	}
	for (a = 0; a < vocab_size; a++) {
		long long id = vocab[a].id_old;
		if (id < vocab_size_old) {
			for (b = 0; b < layer1_size; b++) {
				syn0[a * layer1_size + b] = syn0_old[id * layer1_size + b];
				//syn0[a * layer1_size + b] = (rand() / (real) RAND_MAX - 0.5) / layer1_size;
				//read_counts++;
			}
			//++read_lines;
		} else {
			for (b = 0; b < layer1_size; b++)
			{
				syn0[a * layer1_size + b] = (rand() / (real) RAND_MAX - 0.5) / layer1_size;
				//read_counts++;
			}
			//++read_lines;
		}

	}
	printf("Read %d Lines of Vec File!!\n", read_lines);
	printf("Read %d Counts of Vec File!!\n", read_counts);
	fflush(fin);
	fclose(fin);
	//free(syn0_old);
	//free(vocab_word_point);
}

void InitNegativeSamplingVec(){
	FILE *fin = fopen(model_input_file, "rb");
	unsigned long long next_random = 1;

	int read_lines = 0;
	int read_counts = 0;
	char ch;
	long long a, b;
	long long words, size;
	//char* vocab_word_point1 = (char *) malloc((long long) max_w * sizeof(char));
	char vocab_word_point1[max_w];

	if (fin == NULL) {
		printf("Model file not found\n");
		exit(1);
	}

	//syn1neg_old= (real*) malloc((long long) vocab_size_old * layer1_size * sizeof(real));
	//syn1neg=(real*) malloc((long long) vocab_size * layer1_size * sizeof(real));

	a = posix_memalign((void **)&syn1neg_old, 128, (long long)vocab_size_old * layer1_size * sizeof(real));
	a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));

	if(syn1neg_old == NULL){
		printf("Memory allocation failed\n");
		exit(1);
	}
	if(syn1neg == NULL){
		printf("Memory allocation failed\n");
		exit(1);
	}
	fscanf(fin, "%lld", &words);
	fscanf(fin, "%lld", &size);

	printf("%lld %lld\n",words,size);	

	assert(vocab_size_old == words);
	assert(layer1_size == size);
	//exit(-1);
	if (read_binary) {
		for (a = 0; a < vocab_size_old; a++) {
			fscanf(fin, "%s%c", vocab_word_point1, &ch);
			//printf("%s ",vocab_word_point);
			for (b = 0; b < layer1_size; b++) {
				fread(&syn1neg_old[a * layer1_size + b], sizeof(real), 1,
						fin);
				//printf("%f ",syn1neg_old[a * layer1_size + b]);
				++read_counts;
			}
			//ch = fgetc(fin);
			//printf("\n");
			++read_lines;
		}

	} else {
		long c=0;
		for (a = 0; a < vocab_size_old; a++) {
			//fscanf(fin, "%s%c", vocab_word_point1, &ch);
			c=0;
			while (1) {
				vocab_word_point1[c] = fgetc(fin);
				if (feof(fin) || (vocab_word_point1[c] == ' ')) 
					break;
				if ((c < max_w) && (vocab_word_point1[c] != '\n')) 
					c++;
			}             
			vocab_word_point1[c]='\0';
			//string word=string(vocab_word_point1);
			memset(vocab_word_point1,0,max_w);

			//printf("%s\n",vocab_word_point1);
			for (b = 0; b < layer1_size; b++) {
				fscanf(fin, "%f", &syn1neg_old[a * layer1_size + b]);
				read_counts++;
			}
			ch = fgetc(fin);
			++read_lines;
		}

	}
	for (a = 0; a < vocab_size; a++) {
		long long id = vocab[a].id_old;
		if (id < vocab_size_old) {
			for (b = 0; b < layer1_size; b++) {
				syn1neg[a * layer1_size + b] = syn1neg_old[id * layer1_size + b];
				//read_counts++;
			}
			//++read_lines;
		} else {
			for (b = 0; b < layer1_size; b++) {
				//syn1neg[a * layer1_size + b] = (rand() / (real) RAND_MAX - 0.5) / layer1_size;
				syn1neg[a * layer1_size + b] = 0;
				//read_counts++;
			}
			//++read_lines;
		}
	}

	//exit(-1);
	//free(syn1neg_old);
	//free(vocab_word_point);

	fflush(fin);
	fclose(fin);
	printf("Read %d Lines of Model File!!\n", read_lines);
	printf("Read %d Counts of Model File!!\n", read_counts);

}


void InitNetOnline() {
	InitVec();
	//	exit(-1);
	InitNegativeSamplingVec();
	//exit(-1);
	/*
	   if(hs){
	   a = posix_memalign((void **) &syn1_old, 128,
	   (long long) vocab_size_old * layer1_size * sizeof(real));

	   a = posix_memalign((void **) &syn1, 128,
	   (long long) vocab_size * layer1_size * sizeof(real));

	   if (syn1_old == NULL) {
	   printf("Memory allocation failed\n");
	   exit(1);
	   }

	   fscanf(fin, "%lld", &words);
	   fscanf(fin, "%lld", &size);
	   printf("%lld %lld\n",words,size);
	   vocab_size_old = words;
	   layer1_size=size;

	   if (binary) {
	   for (a = 0; a < vocab_size_old; a++) {
	   fscanf(fin, "%s%c", vocab_word_point, &ch);
	   for (b = 0; b < layer1_size; b++) {
	   fread(&syn1_old[a * layer1_size + b], sizeof(real), 1,
	   fin);
	   ++read_counts;
	   }
	   ch = fgetc(fin);
	   ++read_lines;
	   }

	   for (a = 0; a < vocab_size_old; a++) {
	   fscanf(fin, "%s%c", vocab_word_point, &ch);
	   for (b = 0; b < layer1_size; b++) {
	   fread(&syn0_old[a * layer1_size + b], sizeof(real), 1, fin);
	   ++read_counts;
	   }
	   ch = fgetc(fin);
	   ++read_lines;
	   }
	   } else {
	   for (a = 0; a < vocab_size_old; a++) {
	   fscanf(fin, "%s%c", vocab_word_point, &ch);
	   for (b = 0; b < layer1_size; b++) {
	   fscanf(fin, "%f", &syn1_old[a * layer1_size + b]);
	   read_counts++;
	   }
	   ch = fgetc(fin);
	   ++read_lines;
	   }


	   for (a = 0; a < vocab_size_old; a++) {
	   fscanf(fin, "%s%c", vocab_word_point, &ch);
	   for (b = 0; b < layer1_size; b++) {
	   fscanf(fin, "%f", &syn0_old[a * layer1_size + b]);
	   read_counts++;
	   }
	   ch = fgetc(fin);				//
	   ++read_lines;
	   }
	   }

	   for (a = 0; a < vocab_size; a++) {
	   long long id = vocab[a].id_old;
	   if (id < vocab_size_old) {
	   for (b = 0; b < layer1_size; b++) {
	   syn1[a * layer1_size + b] = syn1_old[id * layer1_size + b];
	   read_counts++;
	   }
	   ++read_lines;
} else {
	for (b = 0; b < layer1_size; b++) {
		syn1[a * layer1_size + b] = 0;
		read_counts++;
	}
	++read_lines;
}
}

for (a = 0; a < vocab_size; a++) {
	long long id = vocab[a].id_old;
	if (id < vocab_size_old) {
		for (b = 0; b < layer1_size; b++) {
			syn0[a * layer1_size + b] = syn0_old[id * layer1_size + b];
			read_counts++;
		}
		++read_lines;
	} else {
		for (b = 0; b < layer1_size; b++)
		{
			syn0[a * layer1_size + b] = (rand() / (real) RAND_MAX - 0.5) / layer1_size;
			read_counts++;
		}
		++read_lines;
	}

}

printf("Read %d Lines of Model File!!\n", read_lines);
printf("Read %d Counts of Model File!!\n", read_counts);
//fclose(fin);
}
*/

//CreateBinaryTree();
//free(vocab_word_point);
printf("Complete CreateBinaryTree!!\n");
}

void SaveModel() {
	long long a, b;

	if (negative <= 0) {
		return;
	}

	FILE *fo = fopen(model_output_file, "wb");//binary format text file

	char* foo_str=model_output_file;
	strcat(foo_str,".txt");

	FILE *foo = fopen(foo_str, "wb");//text format file

	long long number = 1;

	printf("\nlayer1_size: %lld\n", layer1_size);
	printf("Vocab Size: %lld\n", vocab_size);
	printf("Total words: %lld\n",train_words);

	fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
	fprintf(foo, "%lld %lld\n", vocab_size, layer1_size);

	if (save_binary)        //
	{
		if(hs){
			for (a = 0; a < vocab_size; a++) {
				fprintf(fo, "%s ", vocab[a].word);
				fprintf(foo, "%s ", vocab[a].word);
				for (b = 0; b < layer1_size; b++) {
					fwrite(&syn1[a * layer1_size + b], sizeof(real), 1, fo);
					fprintf(foo, "%lf ", syn1[a * layer1_size + b]);
				}
				number++;
				fprintf(fo, "\n");
				fprintf(foo, "\n");
			}
			/*
			   for (a = 0; a < vocab_size; a++) {
			   fprintf(fo, "%s ", vocab[a].word);
			   fprintf(foo, "%s ", vocab[a].word);
			   for (b = 0; b < layer1_size; b++) {
			   fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
			   fprintf(foo, "%lf ", syn0[a * layer1_size + b]);
			   }
			   number++;
			   fprintf(fo, "\n");
			   fprintf(foo, "\n");
			   }
			   */
		}else{
			for (a = 0; a < vocab_size; a++) {
				fprintf(fo, "%s ", vocab[a].word);
				fprintf(foo, "%s ", vocab[a].word);
				for (b = 0; b < layer1_size; b++) {
					fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, fo);
					fprintf(foo, "%lf ", syn1neg[a * layer1_size + b]);
				}
				number++;
				fprintf(fo, "\n");
				fprintf(foo, "\n");
			}
			/*
			   for (a = 0; a < vocab_size; a++) {
			   fprintf(fo, "%s ", vocab[a].word);
			   fprintf(foo, "%s ", vocab[a].word);
			   for (b = 0; b < layer1_size; b++) {
			   fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
			   fprintf(foo, "%lf ", syn0[a * layer1_size + b]);
			   }
			   number++;
			   fprintf(fo, "\n");
			   fprintf(foo, "\n");
			   }
			   */
		}
	} else {
		if(hs){
			for (a = 0; a < vocab_size; a++) {
				fprintf(fo, "%s ", vocab[a].word);
				for (b = 0; b < layer1_size; b++) {
					fprintf(fo, "%lf ", syn1[a * layer1_size + b]);
				}
				number++;
				fprintf(fo, "\n");
			}
			/*
			   for (a = 0; a < vocab_size; a++) {
			   fprintf(fo, "%s ", vocab[a].word);
			   for (b = 0; b < layer1_size; b++) {
			   fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
			   }
			   number++;
			   fprintf(fo, "\n");
			   }
			   */
		}else{
			for (a = 0; a < vocab_size; a++) {
				fprintf(fo, "%s ", vocab[a].word);
				for (b = 0; b < layer1_size; b++) {
					fprintf(fo, "%lf ", syn1neg[a * layer1_size + b]);
				}
				number++;
				fprintf(fo, "\n");
			}
			/*
			   for (a = 0; a < vocab_size; a++) {
			   fprintf(fo, "%s ", vocab[a].word);
			   for (b = 0; b < layer1_size; b++) {
			   fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
			   }
			   number++;
			   fprintf(fo, "\n");
			   }
			   */
		}
	}
	//fprintf(fo, "Read Numbers:%lld", number);
	//fprintf(foo, "Read Numbers:%lld", number);
	fclose(fo);
	fo = NULL;
	fclose(foo);
	foo = NULL;

}
void *TrainModelThread(void *id) {
	long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];

	real sen_alpha[MAX_SENTENCE_LENGTH + 1];
	real word_alpha;

	long long l1, l2, c, target, label;
	unsigned long long next_random = (long long) id;
	real f, g;
	clock_t now;
	real *neu1 = (real *) calloc(layer1_size, sizeof(real));
	real *neu1e = (real *) calloc(layer1_size, sizeof(real));
	FILE *fi = fopen(train_file, "rb");

	if(!fi){
		printf("Open training file failed!\n");
		exit(-1);
	}


	// ??????????每个进程获取一部分
	fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);
	printf("ALL Training Word: %lld\n",train_words);

	/*
	   int i = 0;
	   for(i = 0 ; i < iterations ; ++i){
	   sentence_length = 0, sentence_position = 0;
	   word_count = 0, last_word_count = 0;
	   next_random = (long long) id;
	   fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);
	   word_count_actual =0 ;
	   */
	//exit(-1);
	while (1) {
		if (word_count - last_word_count > 10000){ //Adjust the learning rate per 1W words
			word_count_actual += word_count - last_word_count;
			last_word_count = word_count;
			if ((debug_mode > 1)) {
				now = clock();
				printf(
						"%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ",
						13, alpha,
						word_count_actual / (real) (train_words + 1) * 100,
						word_count_actual
						/ ((real) (now - start + 1)
							/ (real) CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}

			if(update){				
				//alpha = starting_alpha * (1 - word_count_actual / (real) (train_words + 1));
				//alpha = starting_alpha * (1 - (word_count_actual + train_words_old) / (real) (train_words + train_words_old + 1));
				alpha = starting_alpha * (1 - word_count_actual / (real) (train_words + 1));
				for(a = 0 ; a < vocab_size ; ++a){
					vocab[a].alpha=starting_alpha * (1-vocab[a].actual_read/(real)(vocab[a].cn+1));
					if (vocab[a].alpha < starting_alpha * 0.0001)
						vocab[a].alpha = starting_alpha * 0.0001;
				}
				if (alpha < starting_alpha * 0.0001)
					alpha = starting_alpha * 0.0001;
				/*
				   alpha = starting_alpha * (1 - (word_count_actual + train_words_old) / (real) (train_words + train_words_old + 1));
				   if (alpha < starting_alpha * 0.0001)
				   alpha = starting_alpha * 0.0001;
				   */
			}else{
				alpha = starting_alpha * (1 - word_count_actual / (real) (train_words + 1));
				for(a = 0 ; a < vocab_size ; ++a){
					//vocab[a].alpha=vocab[a].alpha * (1-vocab[a].actual_read/(real)(vocab[a].cn+1));
					vocab[a].alpha=starting_alpha * (1-vocab[a].actual_read/(real)(vocab[a].cn+1));
					if (vocab[a].alpha < starting_alpha * 0.0001)
						vocab[a].alpha = starting_alpha * 0.0001;
				}
				if (alpha < starting_alpha * 0.0001)
					alpha = starting_alpha * 0.0001;
			}

		}

		//read a new line
		if (sentence_length == 0) {//read a new sentence
			//printf("New Line!!\n");
			while (1) {
				word = ReadWordIndex(fi);//index of vocab array
				//printf("%lld\n",word);
				//printf("Test1\n");
				//printf("%lld\n",word);
				//printf("%lld %f\n",word,vocab[word].alpha);

				if (word == 0){//the word of <\s> (\n)
					break;
				}
				if (feof(fi)){
					break;
				}
				if (word == -1){
					continue;
				}

				vocab[word].actual_read++;
				word_count++;//sum the all readed words
				if (word == 0){//the word of <\s> (\n)
					break;
				}

				// The subsampling randomly discards frequent words while keeping the ranking same
				if (sample > 0) {
					real ran = (sqrt(vocab[word].cn / (sample * (train_words+train_words_old)))
							+ 1) * (sample * (train_words+train_words_old)) / vocab[word].cn;
					next_random = next_random * (unsigned long long) 25214903917
						+ 11;
					if (ran < (next_random & 0xFFFF) / (real) 65536){
						//printf("4444===>%lld:%f:%lld\t",vocab[word].cn,sample,train_words);
						continue;
					}
				}

				sen[sentence_length] = word;
				//sen_alpha[sentence_length]=vocab[word].alpha;
				sen_alpha[sentence_length]=(vocab[word].alpha + alpha)/2;
				//printf("%f\n",sen_alpha[sentence_length]);


				sentence_length++;
				if (sentence_length >= MAX_SENTENCE_LENGTH)
					break;
			}
			sentence_position = 0;
		}
		if (feof(fi))
			break;

		if (word_count > train_words / num_threads)//only accept the averaged words of train file
			break;

		word = sen[sentence_position];
		//word_alpha = sen_alpha[sentence_position];
		word_alpha = alpha;

		if (word == -1)
			continue;

		//initailize
		for (c = 0; c < layer1_size; c++)
			neu1[c] = 0;

		//initailize
		for (c = 0; c < layer1_size; c++)
			neu1e[c] = 0;

		next_random = next_random * (unsigned long long) 25214903917 + 11;
		b = next_random % window;//a random number between 0 and window-1
		if (cbow) {  //train the cbow architecture
			// in -> hidden     a:    b<-->windows<-->2*window+1-b
			for (a = b; a < window * 2 + 1 - b; a++)
				if (a != window) {
					c = sentence_position + a - window ;
					if (c < 0)
						continue;
					if (c >= sentence_length)
						continue;

					last_word = sen[c];//

					if (last_word == -1)
						continue;
					for (c = 0; c < layer1_size; c++)
						neu1[c] += syn0[c + last_word * layer1_size];
				}

			if (hs)
				for (d = 0; d < vocab[word].codelen; d++) {
					f = 0;
					l2 = vocab[word].point[d] * layer1_size;
					// Propagate hidden -> output
					for (c = 0; c < layer1_size; c++)
						f += neu1[c] * syn1[c + l2];
					if (f <= -MAX_EXP)
						continue;
					else if (f >= MAX_EXP)
						continue;
					else
						f = expTable[(int) ((f + MAX_EXP)
								* (EXP_TABLE_SIZE / MAX_EXP / 2))];
					// 'g' is the gradient multiplied by the learning rate
					g = (1 - vocab[word].code[d] - f) * word_alpha;
					// Propagate errors output -> hidden
					for (c = 0; c < layer1_size; c++)
						neu1e[c] += g * syn1[c + l2];
					// Learn weights hidden -> output
					for (c = 0; c < layer1_size; c++)
						syn1[c + l2] += g * neu1[c];
				}//end for hs

			// NEGATIVE SAMPLING
			if (negative > 0)
				for (d = 0; d < negative + 1; d++) {
					if (d == 0) {
						target = word;
						label = 1;
					} else {
						//fixed the next_random

						next_random = next_random
							* (unsigned long long) 25214903917 + 11;
						target = table[(next_random >> 16) % table_size];

						if (target == 0)
							target = next_random % (vocab_size - 1) + 1;
						if (target == word)
							continue;
						label = 0;
					}
					l2 = target * layer1_size;
					f = 0;
					for (c = 0; c < layer1_size; c++)
						f += neu1[c] * syn1neg[c + l2];
					if (f > MAX_EXP)
						g = (label - 1) * word_alpha;
					else if (f < -MAX_EXP)
						g = (label - 0) * word_alpha;
					else
						g = (label
								- expTable[(int) ((f + MAX_EXP)
									* (EXP_TABLE_SIZE / MAX_EXP / 2))])
							* word_alpha;
					for (c = 0; c < layer1_size; c++)
						neu1e[c] += g * syn1neg[c + l2];
					for (c = 0; c < layer1_size; c++)
						syn1neg[c + l2] += g * neu1[c];
				}//end for negative sampling
			// hidden -> in
			for (a = b; a < window * 2 + 1 - b; a++)
				if (a != window) {
					c = sentence_position - window + a;
					if (c < 0)
						continue;
					if (c >= sentence_length)
						continue;
					last_word = sen[c];

					if (last_word == -1)
						continue;
					for (c = 0; c < layer1_size; c++)
						syn0[c + last_word * layer1_size] += neu1e[c];
				}
		} else {  //train skip-gram
			for (a = b; a < window * 2 + 1 - b; a++)
				if (a != window) {
					c = sentence_position - window + a;
					if (c < 0)
						continue;
					if (c >= sentence_length)
						continue;
					last_word = sen[c];

					if (last_word == -1)
						continue;
					l1 = last_word * layer1_size;
					for (c = 0; c < layer1_size; c++)
						neu1e[c] = 0;
					// HIERARCHICAL SOFTMAX
					if (hs)
						for (d = 0; d < vocab[word].codelen; d++) {
							f = 0;
							l2 = vocab[word].point[d] * layer1_size;//index of dth word in array
							// Propagate hidden -> output
							for (c = 0; c < layer1_size; c++)
								f += syn0[c + l1] * syn1[c + l2];
							if (f <= -MAX_EXP)
								continue;
							else if (f >= MAX_EXP)
								continue;
							else
								f = expTable[(int) ((f + MAX_EXP)
										* (EXP_TABLE_SIZE / MAX_EXP / 2))];
							// 'g' is the gradient multiplied by the learning rate
							//g = (1 - vocab[word].code[d] - f) * alpha;
							g = (1 - vocab[word].code[d] - f) * word_alpha;
							// Propagate errors output -> hidden
							for (c = 0; c < layer1_size; c++)
								neu1e[c] += g * syn1[c + l2];
							// Learn weights hidden -> output
							for (c = 0; c < layer1_size; c++)
								syn1[c + l2] += g * syn0[c + l1];
						}//end for hs
					// NEGATIVE SAMPLING
					if (negative > 0)
						for (d = 0; d < negative + 1; d++) {//sampling negative+1 words,include current word itself
							if (d == 0) {//
								target = word;
								label = 1;
							} else {

								next_random = next_random
									* (unsigned long long) 25214903917 + 11;
								target =
									table[(next_random >> 16) % table_size];

								if (target == 0){//skip the <\s> word{
									target = next_random % (vocab_size - 1) + 1;
								}
								if (target == word)//skip the current word
									continue;
								label = 0;
								}
								l2 = target * layer1_size;

								f = 0;
								for (c = 0; c < layer1_size; c++)
									f += syn0[c + l1] * syn1neg[c + l2];
								if (f > MAX_EXP)
									//g = (label - 1) * alpha;
									g = (label - 1) * word_alpha;
								else if (f < -MAX_EXP)
									//g = (label - 0) * alpha;
									g = (label - 0) * word_alpha;
								else
									/*
									   g =
									   (label
									   - expTable[(int) ((f + MAX_EXP)
									 * (EXP_TABLE_SIZE
									 / MAX_EXP / 2))])
									 * alpha;
									 */
									g =
										(label- expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * word_alpha;

								//printf("%f\n",g);
								for (c = 0; c < layer1_size; c++)
									neu1e[c] += g * syn1neg[c + l2];
								for (c = 0; c < layer1_size; c++){
									syn1neg[c + l2] += g * syn0[c + l1];
								}


							}//end for negative sampling

							// Learn weights input -> hidden
							for (c = 0; c < layer1_size; c++){
								//printf("%f\n",neu1e[c]);
								syn0[c + l1] += neu1e[c];
							}
						}//end for traverse one word
				}//end for ski-gram

			sentence_position++;//traverse one word's windows
			if (sentence_position >= sentence_length) {
				sentence_length = 0;
				continue;
			}
		}


		fclose(fi);
		free(neu1);
		free(neu1e);
		pthread_exit(NULL);
	}

	void save_word_vector(){
		long long a,b;
		FILE *fo,*fo_other;
		fo = fopen(output_file, "wb");
		fo_other = fopen(output_file_other, "w");
		fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
		fprintf(fo_other, "%lld %lld\n", vocab_size, layer1_size);
		for (a = 0; a < vocab_size; a++) {
			fprintf(fo, "%s ", vocab[a].word);
			fprintf(fo_other, "%s ", vocab[a].word);
			if (save_binary == 1)
				for (b = 0; b < layer1_size; b++)
					fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
			else if(save_binary == 0)
				for (b = 0; b < layer1_size; b++)
					fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
			else if(save_binary == 2){
				for (b = 0; b < layer1_size; b++)
					fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
				fprintf(fo, "\n");
				for (b = 0; b < layer1_size; b++) 
					fprintf(fo_other, "%lf ", syn0[a * layer1_size + b]);		
				fprintf(fo_other, "\n");
			}
		}
		fclose(fo);
		fclose(fo_other);
	} 



	void TrainModel() {
		long long a, b, c, d;
		FILE *fo,*fo_other;
		int i=0;
		pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
		if(!pt){
			printf("Allocate pthread failed!\n");
		}
		printf("Starting training using file %s\n", train_file);
		starting_alpha = alpha;
		if (update) {
			if (read_vocab_file[0] == 0) {
				printf("Vocabulary file not found\n");
				//exit(1);
			}
			ReadVocab_Update();
			LearnVocabFromTrainFile_Update();
			InitNetOnline();


			//save_word_vector();
			//SaveModel();
			//exit(-1);
			//starting_alpha = starting_alpha * (1 - (word_count_actual + train_words_old) / (real) (train_words + train_words_old + 1));
			printf("Complete InitNetOnline!!\n");
		} else {
			if (read_vocab_file[0] != 0)
				ReadVocab();
			else
				LearnVocabFromTrainFile();
			InitNet();
		}

		if (save_vocab_file[0] != 0){
			//	printf("Save vocab file: %s\n",save_vocab_file);
			SaveVocab();
			printf("Save vocab file completed!!");
		}

		if (output_file[0] == 0){
			printf("Not define out file name!!\n");
			return;
		}

		if (negative > 0)
			InitUnigramTable();

		//exit(-1);
		printf("Complete InitUnigramTable!!\n");
		start = clock();

		//exit(-1);
		for(i = 0 ; i < iterations ; ++i){
			word_count_actual=0;
			for (a = 0; a < num_threads; a++)
				pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
			for (a = 0; a < num_threads; a++)
				pthread_join(pt[a], NULL);
		}
		fo = fopen(output_file, "wb");
		fo_other = fopen(output_file_other, "w");
		if (classes == 0) {
			// Save the word vectors
			save_word_vector();
		} else {
			// Run K-means on the word vectors
			int clcn = classes, iter = 10, closeid;
			int *centcn = (int *) malloc(classes * sizeof(int));
			int *cl = (int *) calloc(vocab_size, sizeof(int));
			real closev, x;
			real *cent = (real *) calloc(classes * layer1_size, sizeof(real));
			for (a = 0; a < vocab_size; a++)
				cl[a] = a % clcn;
			for (a = 0; a < iter; a++) {
				for (b = 0; b < clcn * layer1_size; b++)
					cent[b] = 0;
				for (b = 0; b < clcn; b++)
					centcn[b] = 1;
				for (c = 0; c < vocab_size; c++) {
					for (d = 0; d < layer1_size; d++)
						cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
					centcn[cl[c]]++;
				}
				for (b = 0; b < clcn; b++) {
					closev = 0;
					for (c = 0; c < layer1_size; c++) {
						cent[layer1_size * b + c] /= centcn[b];
						closev += cent[layer1_size * b + c]
							* cent[layer1_size * b + c];
					}
					closev = sqrt(closev);
					for (c = 0; c < layer1_size; c++)
						cent[layer1_size * b + c] /= closev;
				}
				for (c = 0; c < vocab_size; c++) {
					closev = -10;
					closeid = 0;
					for (d = 0; d < clcn; d++) {
						x = 0;
						for (b = 0; b < layer1_size; b++)
							x += cent[layer1_size * d + b]
								* syn0[c * layer1_size + b];
						if (x > closev) {
							closev = x;
							closeid = d;
						}
					}
					cl[c] = closeid;
				}
			}
			// Save the K-means classes
			for (a = 0; a < vocab_size; a++)
				fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
			free(centcn);
			free(cent);
			free(cl);
		}
		fclose(fo);
	}

	int ArgPos(char *str, int argc, char **argv) {
		int a;
		for (a = 1; a < argc; a++)
			if (!strcmp(str, argv[a])) {
				if (a == argc - 1) {
					printf("Argument missing for %s\n", str);
					exit(1);
				}
				return a;
			}
		return -1;
	}

	int main(int argc, char **argv) {
		int i;
		if (argc == 1) {
			printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
			printf("Options:\n");
			printf("Parameters for training:\n");
			printf("\t-train <file>\n");
			printf("\t\tUse text data from <file> to train the model\n");
			printf("\t-output <file>\n");
			printf(
					"\t\tUse <file> to save the resulting word vectors / word clusters\n");
			printf("\t-size <int>\n");
			printf("\t\tSet size of word vectors; default is 100\n");
			printf("\t-window <int>\n");
			printf("\t\tSet max skip length between words; default is 5\n");
			printf("\t-sample <float>\n");
			printf(
					"\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
			printf(
					" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
			printf("\t-hs <int>\n");
			printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
			printf("\t-negative <int>\n");
			printf(
					"\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
			printf("\t-threads <int>\n");
			printf("\t\tUse <int> threads (default 1)\n");
			printf("\t-min-count <int>\n");
			printf(
					"\t\tThis will discard words that appear less than <int> times; default is 5\n");
			printf("\t-alpha <float>\n");
			printf("\t\tSet the starting learning rate; default is 0.025\n");
			printf("\t-classes <int>\n");
			printf(
					"\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
			printf("\t-debug <int>\n");
			printf(
					"\t\tSet the debug mode (default = 2 = more info during training)\n");
			printf("\t-binary <int>\n");
			printf(
					"\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
			printf("\t-save-vocab <file>\n");
			printf("\t\tThe vocabulary will be saved to <file>\n");
			printf("\t-read-vocab <file>\n");
			printf(
					"\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
			printf("\t-cbow <int>\n");
			printf(
					"\t\tUse the continuous bag of words model; default is 0 (skip-gram model)\n");
			printf("\nExamples:\n");
			printf(
					"./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
			return 0;
		}
		output_file[0] = 0;
		output_file_other[0] = 0;
		save_vocab_file[0] = 0;
		read_vocab_file[0] = 0;
		model_input_file[0] = 0;
		model_output_file[0] = 0;
		vec_input_file[0] = 0;
		if ((i = ArgPos((char *) "-size", argc, argv)) > 0)
			layer1_size = atoi(argv[i + 1]);
		if ((i = ArgPos((char *) "-train", argc, argv)) > 0)
			strcpy(train_file, argv[i + 1]);
		if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0)
			strcpy(save_vocab_file, argv[i + 1]);
		if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0)
			strcpy(read_vocab_file, argv[i + 1]);
		if ((i = ArgPos((char *) "-debug", argc, argv)) > 0)
			debug_mode = atoi(argv[i + 1]);
		if ((i = ArgPos((char *) "-read-binary", argc, argv)) > 0)
			read_binary = atoi(argv[i + 1]);
		if ((i = ArgPos((char *) "-save-binary", argc, argv)) > 0)
			save_binary = atoi(argv[i + 1]);
		if ((i = ArgPos((char *) "-cbow", argc, argv)) > 0)
			if ((i = ArgPos((char *) "-cbow", argc, argv)) > 0)
				cbow = atoi(argv[i + 1]);
		if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0)
			alpha = atof(argv[i + 1]);
		if ((i = ArgPos((char *) "-output", argc, argv)) > 0){
			strcpy(output_file, argv[i + 1]);
			strcpy(output_file_other,argv[i+1]);
			char *suffix=".txt";
			strcat(output_file_other,suffix);
		}
		if ((i = ArgPos((char *) "-window", argc, argv)) > 0)
			window = atoi(argv[i + 1]);
		if ((i = ArgPos((char *) "-sample", argc, argv)) > 0)
			sample = atof(argv[i + 1]);
		if ((i = ArgPos((char *) "-hs", argc, argv)) > 0)
			hs = atoi(argv[i + 1]);
		if ((i = ArgPos((char *) "-negative", argc, argv)) > 0)
			negative = atoi(argv[i + 1]);
		if ((i = ArgPos((char *) "-threads", argc, argv)) > 0)
			num_threads = atoi(argv[i + 1]);
		if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0)
			min_count = atoi(argv[i + 1]);
		if ((i = ArgPos((char *) "-classes", argc, argv)) > 0)
			classes = atoi(argv[i + 1]);
		if ((i = ArgPos((char *) "-model-output", argc, argv)) > 0)
			strcpy(model_output_file, argv[i + 1]);
		if ((i = ArgPos((char *) "-model-input", argc, argv)) > 0)
			strcpy(model_input_file, argv[i + 1]);
		if ((i = ArgPos((char *) "-vec-input-file", argc, argv)) > 0)
			strcpy(vec_input_file, argv[i + 1]);
		if ((i = ArgPos((char *) "-update", argc, argv)) > 0)
			update = atoi(argv[i + 1]);
		if ((i = ArgPos((char *) "-iterations", argc, argv)) > 0)
			iterations = atoi(argv[i + 1]);


		vocab = (struct vocab_word *) calloc(vocab_max_size,
				sizeof(struct vocab_word));
		if(!vocab){
			printf("Allocate vocab mem failed!\n");
		}
		vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
		expTable = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
		for (i = 0; i < EXP_TABLE_SIZE; i++) {
			expTable[i] = exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
			expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
		}
		if(output_file[0] != 0){
			printf("Output Vector File Path:%s\n",output_file);
		}
		if(output_file_other[0] != 0){
			printf("Output Vector File Of Text Format Path:%s\n",output_file_other);
		}
		if(save_vocab_file[0] != 0){
			printf("Save vocab file path:%s\n",save_vocab_file);
		}
		if(read_vocab_file[0] != 0){
			printf("Read vocab file path:%s\n",read_vocab_file);
		}
		if(model_input_file[0] != 0){
			printf("Model input file path:%s\n",model_input_file);
		}
		if(model_output_file[0] != 0){
			printf("Model output file path:%s\n",model_output_file);
		}
		TrainModel();
		printf("Complete training,Then output model file to:%s\n", model_output_file);
		//		printf("Input Model File:%s\n", model_input_file);
		SaveModel();
		return 0;
	}
