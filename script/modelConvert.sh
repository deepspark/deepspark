#!/bin/bash
if [ "$#" -eq 3 ]
then
for name in `ls $1/*.weight`
do
	modelName=${name//weight/caffemodel}
	java -cp deepspark-0.0.1-SNAPSHOT-jar-with-dependencies.jar org.acl.deepspark.utils.WeightConverter $name $modelName $2 $3	
done
else
	echo "usage: ./modelConvert.sh <target dir> <caffe solver prototxt> <caffe net prototxt>";
fi
