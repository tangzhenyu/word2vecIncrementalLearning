#!/bin/sh
hive -e "
add jar /data0/user/zhangkan/Word2Vec/sh/ForWord2VecDataInput.jar;
create temporary function getTerm as 'UDF.ForWord2VecDataInput';
insert overwrite local directory './tempfile'
select getTerm(title_wordseg), getTerm(content_wordseg) from ods_sinaportal_url_content_wordseg;
"
cat ./tempfile/*>tempfile.txt
java -jar multiline2oneline.jar -input=./tempfile.txt -output=data 
