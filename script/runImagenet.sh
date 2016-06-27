#!/bin/bash

if [ "$#" -eq 3 ]
then
	spark-submit --master yarn-cluster --files ../ext/libcaffeext.so,$2,googlenet_net.prototxt,googlenet_solver.prototxt,mean.binaryproto --num-executors $1 --class org.acl.deepspark.nn.driver.AsyncEASGDImageNetTest deepspark-0.0.1-SNAPSHOT-jar-with-dependencies.jar googlenet_solver.prototxt googlenet_net.prototxt `basename $2`
	mkdir $3
	hdfs dfs -get imagenet_test/* $3;
	hdfs dfs -rm -r imagenet_test;
else
	echo "runImagenet.sh <# of node> <deepspark.prototxt> <save dir>";
fi
