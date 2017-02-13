make clean
make

iterations1=1
dim1=600
windows1=5
threads1=35
input_file1="/data/tangzhenyu/all_words_content_disturb_order"
output_file1=${input_file1}"_vector_"${iterations1}"_"${dim1}"_"${windows1}"_increment_alpha_2.bin"
save_vocab_file1=${input_file1}"_words_"${iterations1}"_"${dim1}"_"${windows1}"_increment_alpha_2.vocab"
model_output_file1=${input_file1}"_model_"${iterations1}"_"${dim1}"_"${windows1}"_increment_alpha_2.model"

iterations2=1
dim2=600
windows2=5
threads2=35
model_input_file2=${model_output_file1}
read_vocab_file2=${save_vocab_file1}
input_file2="/data/tangzhenyu/sports_words_content"
output_file2=${input_file2}"_vector_"${iterations2}"_"${dim2}"_"${windows2}".bin"
save_vocab_file2=${input_file2}"_words_"${iterations2}"_"${dim2}"_"${windows2}".vocab"
model_output_file2=${input_file2}"_model_"${iterations2}"_"${dim2}"_"${windows2}".model"
command="./word2vec -train ${input_file1} -output ${output_file1} -model-output ${model_output_file1} -cbow 0 -size ${dim1} -window 5 -negative 8 -hs 0 -sample 1e-5 -threads ${threads1} -binary 2 -iterations ${iterations1} -update 0 -save-vocab ${save_vocab_file1} -min-count 1"
echo ${command}
${command}
#./word2vec -train ${input_file2} -output ${output_file2} -model-output ${model_output_file2} -cbow 0 -size ${dim2} -window 5 -negative 8 -hs 0 -sample 1e-5 -threads 35 -binary 2 -iterations ${iterations2} -update 1 -read-vocab ${read_vocab_file2} -save-vocab ${save_vocab_file2} -model-input ${model_input_file2} -min-count 100 
#./word2vec -train ${input_file1} -output ${output_file1} -model-output ${model_output_file1} -cbow 0 -size ${dim1} -window 5 -hs 1 -sample 1e-5 -threads 35 -binary 2 -iterations ${iterations1} -update 0 -save-vocab ${save_vocab_file1} -min-count 1000
#./word2vec -train ${input_file2} -output ${output_file2} -model-output ${model_output_file2} -cbow 0 -size ${dim2} -window 5 -hs 1 -sample 1e-5 -threads 35 -binary 2 -iterations ${iterations2} -update 1 -read-vocab ${read_vocab_file2} -save-vocab ${save_vocab_file2} -model-input ${model_input_file2} -min-count 100 
