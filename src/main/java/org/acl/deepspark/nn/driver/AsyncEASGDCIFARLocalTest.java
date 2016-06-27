package org.acl.deepspark.nn.driver;

import java.io.File;
import java.net.InetAddress;
import java.util.Arrays;
import java.util.Date;

import org.acl.deepspark.data.LMDBWriter;
import org.acl.deepspark.data.Record;
import org.acl.deepspark.nn.DistNet;
import org.acl.deepspark.nn.async.easgd.DistAsyncEASGDSolver;
import org.acl.deepspark.utils.CIFARLoader;
import org.acl.deepspark.utils.DeepSparkParamLoader;
import org.acl.deepspark.utils.DeepSparkConf.DeepSparkParam;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

/** 
 * Created by Hanjoo on 2016-01-13.
 */
public class AsyncEASGDCIFARLocalTest {
    public static void main(String[] args) throws Exception {
    	SparkConf conf = new SparkConf().setAppName("AsyncCIFARTest")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        
        JavaSparkContext sc = new JavaSparkContext(conf);
        
        DeepSparkParam param = DeepSparkParamLoader.readConf(args[2]);
        System.out.println("Data Loading...");
        Record[] train_sample = CIFARLoader.loadFromHDFSText("/data/CIFAR-10/train_batch.txt", false);
        Record[] test_sample = CIFARLoader.loadFromHDFSText("/data/CIFAR-10/test_batch.txt", false);

        int numExecutors = conf.getInt("spark.executor.instances", -1);
        System.out.println("number of executors = " + numExecutors);
        JavaRDD<Record> train_data,test_data;
        if(numExecutors != -1) {
        	train_data = sc.parallelize(Arrays.asList(train_sample), numExecutors).cache();
        	test_data= sc.parallelize(Arrays.asList(test_sample), numExecutors).cache();
        } else {
        	train_data = sc.parallelize(Arrays.asList(train_sample)).cache();
        	test_data= sc.parallelize(Arrays.asList(test_sample)).cache();
        }
        
        File trainFile = new File("cifar10_train_lmdb");
        trainFile.mkdirs();
        File testFile = new File("cifar10_test_lmdb");
        testFile.mkdirs();
        
        LMDBWriter trainWriter = new LMDBWriter("cifar10_train_lmdb");
        for(int i = 0; i < train_sample.length; i++) {
        	trainWriter.putSample(train_sample[i]);
        }
        trainWriter.closeLMDB();
        
        LMDBWriter testWriter = new LMDBWriter("cifar10_test_lmdb");
        for(int i = 0; i < test_sample.length; i++) {
        	testWriter.putSample(test_sample[i]);
        }
        testWriter.closeLMDB();
        
        Integer[] tempkey = new Integer[numExecutors];
        for(int i = 0; i < tempkey.length; i++)
        	tempkey[i] = i;
        
        JavaRDD<Integer> rddkey = sc.parallelize(Arrays.asList(tempkey), numExecutors).cache();
        
        DistNet net = new DistNet(args[0], args[1]);
        
        String serverHost = InetAddress.getLocalHost().getHostAddress();
        final int port = 10020;
        System.out.println("ParameterServer host: " + serverHost);
        DistAsyncEASGDSolver driver = new DistAsyncEASGDSolver(net, serverHost, port, param);
        
        driver.prepareLocal(train_data, "cifar10_train_lmdb", numExecutors);
        driver.prepareLocal(test_data, "cifar10_test_lmdb", numExecutors);
        
        System.out.println("Start Learning...");
        Date startTime = new Date();
        
        driver.trainWithLMDB(rddkey, null, numExecutors);
        sc.close();
        Date endTime = new Date();
        
        driver.saveWeight("final_cifar.weight");
        // float[] a = driver.testWithLMDB();
        long time = endTime.getTime() - startTime.getTime();
        System.out.println(String.format("Training time: %f secs", (double) time / 1000));
        //System.out.println(String.format("Accuracy: %f ", a[0]));
    }
}
