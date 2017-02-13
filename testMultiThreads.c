#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define WORDS 1000000
const int num_threads = 40;
int  word_count_actual = 0;


long long words[WORDS];

static void * TrainModelThread(void *id)
{
	int i;
	int j = 0;
	/*
	   while(j < 1000){
	   continue;
	   }
	   */
	for(j = 0 ; j < 100 ; ++j)
		for(i = 0 ; i < WORDS ; ++i){
			words[i]++;	
		}
}

//	long id1 = long(id);

int main(int argc, char **argv) 
{
	long a;
	int i = 0 ;
	for(i = 0 ; i < WORDS ; ++i){
		words[i]=0;
	}
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	long long sum=0;
	for(i = 0 ; i < WORDS ; ++i){
		//printf("%lld\n",words[i]);
		sum += words[i];
	}
	printf("%lld\n",sum);
}
