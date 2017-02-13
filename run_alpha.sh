make clean
make

iterations1=1
dim1=600
windows1=5
input_file1="/data/tangzhenyu/base_words_content"
output_file1=${input_file1}"_vector_"${iterations1}"_"${dim1}"_"${windows1}"_increment_alpha.bin"
save_vocab_file1=${input_file1}"_words_"${iterations1}"_"${dim1}"_"${windows1}"_increment_alpha.vocab"
model_output_file1=${input_file1}"_model_"${iterations1}"_"${dim1}"_"${windows1}"_increment_alpha.model"

iterations2=2
dim2=600
windows2=5
binary2=0
threads2=35
vec_input_file2=${output_file1}".txt"
model_input_file2=${model_output_file1}".txt"
read_vocab_file2=${save_vocab_file1}
input_file2="/data/tangzhenyu/sports_words_content"
output_file2=${input_file2}"_vector_"${iterations2}"_"${dim2}"_"${windows2}"_increment_alpha.bin"
save_vocab_file2=${input_file2}"_words_"${iterations2}"_"${dim2}"_"${windows2}"_increment_alpha.vocab"
model_output_file2=${input_file2}"_model_"${iterations2}"_"${dim2}"_"${windows2}"_increment_alpha.model"
#./word2vec -train ${input_file1} -output ${output_file1} -model-output ${model_output_file1} -cbow 0 -size ${dim1} -window 5 -negative 8 -hs 0 -sample 1e-5 -threads 35 -binary 2 -iterations ${iterations1} -update 0 -save-vocab ${save_vocab_file1} -min-count 5
command="./word2vec -train ${input_file2} -output ${output_file2} -model-output ${model_output_file2} -cbow 0 -size ${dim2} -window 5 -negative 8 -hs 0 -sample 1e-5 -threads ${threads2} -binary ${binary2} -iterations ${iterations2} -update 1 -read-vocab ${read_vocab_file2} -save-vocab ${save_vocab_file2} -model-input ${model_input_file2} -min-count 1 -vec-input-file ${vec_input_file2}"
echo ${command}
${command}

#./word2vec -train ${input_file1} -output ${output_file1} -model-output ${model_output_file1} -cbow 0 -size ${dim1} -window 5 -hs 1 -sample 1e-5 -threads 35 -binary 2 -iterations ${iterations1} -update 0 -save-vocab ${save_vocab_file1} -min-count 1000
#./word2vec -train ${input_file2} -output ${output_file2} -model-output ${model_output_file2} -cbow 0 -size ${dim2} -window 5 -hs 1 -sample 1e-5 -threads 35 -binary 2 -iterations ${iterations2} -update 1 -read-vocab ${read_vocab_file2} -save-vocab ${save_vocab_file2} -model-input ${model_input_file2} -min-count 100 
