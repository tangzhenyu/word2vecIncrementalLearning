make clean
make
input_file="/data/tangzhenyu/sports_words_content"
iterations=1
dim=600
windows=5
output_file="/data/tangzhenyu/sports_words_content_vector_"${iterations}"_"${dim}"_"${windows}".bin"
model_output_file="/data/tangzhenyu/sports_words_model_"${iterations}"_"${dim}"_"${windows}".txt"
#./word2vec -train /data/tangzhenyu/all_words_content -output all_words_content_vector.bin -model-output model.out11 -cbow 0 -size 600 -window 5 -negative 8 -hs 0 -sample 1e-3 -threads 20 -binary 2 -iterations 3
./word2vec -train ${input_file} -output ${output_file} -model-output ${model_output_file} -cbow 0 -size ${dim} -window ${windows} -hs 1 -threads 8 -binary 2 -iterations ${iterations} -update 0
