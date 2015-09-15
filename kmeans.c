#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

int main(int argc, char **argv) {
const long long max_size = 2000;
const long long max_w = 50; 
//const int layer1_size = 400;
FILE *f;
char train_file[max_size];
f = fopen("/data/zhangliping/word2vec/data/new_fenci/fenci_result/merge750_oneline_neg_item1.bin", "rb");
long long words, layer1_size;
float *syn0;
char *vocab;
char ch;
int a,b,c,d;

printf("begin:\n");
fscanf(f, "%lld", &words);
fscanf(f, "%lld", &layer1_size);
printf("%lld %lld\n",words,layer1_size);

vocab = (char *)malloc((long long)words * max_w * sizeof(char));
syn0 = (float *)malloc((long long)words * (long long)layer1_size * sizeof(float));
for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &vocab[b * max_w], &ch);
    for (a = 0; a < layer1_size; a++) fread(&syn0[a + b * layer1_size], sizeof(float), 1, f);
    float len = 0;
    for (a = 0; a < layer1_size; a++) len += syn0[a + b * layer1_size] * syn0[a + b * layer1_size];
    len = sqrt(len);
    for (a = 0; a < layer1_size; a++) syn0[a + b * layer1_size] /= len;
}
fclose(f);

int vocab_size = words;
int classes = 500;
int clcn = classes, iter = 10, closeid;
int *centcn = (int *)malloc(classes * sizeof(int));
int *cl = (int *)calloc(vocab_size, sizeof(int));
float closev, x;
float *cent = (float *)calloc(classes * layer1_size, sizeof(float));
for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
for (a = 0; a < iter; a++) {
    for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
    for (b = 0; b < clcn; b++) centcn[b] = 1;
    for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) {
            cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
            centcn[cl[c]]++;
        }
    }
    for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
            cent[layer1_size * b + c] /= centcn[b];
            closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
    }
    for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
            x = 0;
            for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
            if (x > closev) {
                closev = x;
                closeid = d;
            }
        }
        cl[c] = closeid;
    }
}
for (a = 0; a < vocab_size; a++)
{
    printf("%s\t", &vocab[a*max_w]);
    printf("%d\n", cl[a]);
}
free(centcn);
free(cent);
free(cl);
return 0;
}



