#!/bin/sh
hive -e "
add jar /data0/user/zhangkan/Word2Vec/sh/ForWord2VecDataInput.jar;
create temporary function getTerm as 'UDF.ForWord2VecDataInput';
insert overwrite local directory './tempfile'
select getTerm(XX), getTerm(YY) from XXXX where XXXXXX;
"
cat ./tempfile/*>tempfile.txt
java -jar multiline2oneline.jar -input=./tempfile.txt -output=data 
