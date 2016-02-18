#!/bin/bash

if [ "$#" -eq 3 ]
then
	spark-submit --master yarn-cluster --files libcaffeext.so,$2,cifar10_deepspark.prototxt,cifar10_128.prototxt,mean.binaryproto --num-executors $1 --class org.acl.deepspark.nn.driver.AsyncEASGDCIFARLocalTest deepspark-0.0.1-SNAPSHOT-jar-with-dependencies.jar cifar10_deepspark.prototxt cifar10_128.prototxt `basename $2`
	mkdir $3
	hdfs dfs -get cifar_test/* $3;
	hdfs dfs -rm -r cifar_test
else
	echo "runCIFAR.sh <# of node> <deepspark.prototxt> <save dir>";
fi
