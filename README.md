# DeepSpark
Deeplearning framework running on Spark

# Requires
* Apache Hadoop YARN and Spark
* Caffe prerequisites for Worker nodes (http://caffe.berkeleyvision.org/installation.html)
* CUDA
* cuDNN
* protobuf, glog, gflags

# Build
<code> $ mvn package </code>

# running demos
Sample running scripts are on scripts/
* runCIFAR.sh: CIFAR10 demo
* runImageNet.sh: imagenet demo
* modelConvert.sh: convert DeepSpark weight to caffemodel
