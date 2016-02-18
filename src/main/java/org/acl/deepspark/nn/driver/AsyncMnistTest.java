package org.acl.deepspark.nn.driver;

import java.net.InetAddress;
import java.util.Arrays;
import java.util.Date;

import org.acl.deepspark.data.Record;
import org.acl.deepspark.nn.DistNet;
import org.acl.deepspark.utils.MnistLoader;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

/**
 * Created by Jaehong on 2015-10-05.
 */
public class AsyncMnistTest {
    public static void main(String[] args) throws Exception {
    	SparkConf conf = new SparkConf().setAppName("AsyncMnistTest")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        
        JavaSparkContext sc = new JavaSparkContext(conf);

        System.out.println("Data Loading...");
        Record[] train_sample = MnistLoader.loadFromHDFS("/data/MNIST/mnist_train.txt", true);
        Record[] test_sample = MnistLoader.loadFromHDFS("/data/MNIST/mnist_test.txt", true);

        int numExecutors = conf.getInt("spark.executor.instances", -1);
        System.out.println("number of executors = " + numExecutors);
        JavaRDD<Record> train_data;
        if(numExecutors != -1)        	
        	train_data = sc.parallelize(Arrays.asList(train_sample), numExecutors).cache();
        else
        	train_data = sc.parallelize(Arrays.asList(train_sample)).cache();
        
        DistNet net = new DistNet(args[0], args[1]);

        String serverHost = InetAddress.getLocalHost().getHostAddress();
        final int[] port = new int[] {10020, 10021};
        System.out.println("ParameterServer host: " + serverHost);
        DistAsyncNeuralNetRunner driver = new DistAsyncNeuralNetRunner(net, serverHost, port, 10);
        driver.test(Arrays.asList(test_sample));
        
        System.out.println("Start Learning...");
        Date startTime = new Date();
        driver.train(train_data);
        sc.close();
        
        Date endTime = new Date();
        float[] ret = driver.test(Arrays.asList(test_sample));
        
        long time = endTime.getTime() - startTime.getTime();
        System.out.println(String.format("Training time: %f secs", (double) time / 1000));
        System.out.println(String.format("Accuracy: %f ", ret[0]));
    }
}
