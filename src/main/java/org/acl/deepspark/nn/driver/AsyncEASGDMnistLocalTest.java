package org.acl.deepspark.nn.driver;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.InetAddress;
import java.util.Arrays;
import java.util.Date;

import org.acl.deepspark.data.LMDBWriter;
import org.acl.deepspark.data.Record;
import org.acl.deepspark.nn.DistNet;
import org.acl.deepspark.nn.async.easgd.DistAsyncEASGDSolver;
import org.acl.deepspark.utils.DeepSparkParamLoader;
import org.acl.deepspark.utils.MnistLoader;
import org.acl.deepspark.utils.DeepSparkConf.DeepSparkParam;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;

/**
 * Created by Hanjoo on 2016-01-18.
 */
public class AsyncEASGDMnistLocalTest {
    public static void main(String[] args) throws Exception {
    	SparkConf conf = new SparkConf().setAppName("AsyncMnistTest")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        
        JavaSparkContext sc = new JavaSparkContext(conf);

        DeepSparkParam param = DeepSparkParamLoader.readConf(args[2]);
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
        
        
        Integer[] tempkey = new Integer[numExecutors];
        for(int i = 0; i < tempkey.length; i++)
        	tempkey[i] = i;
        
        JavaRDD<Integer> rddkey = sc.parallelize(Arrays.asList(tempkey), numExecutors).cache();
        rddkey.foreach(new VoidFunction<Integer>() {
			
			@Override
			public void call(Integer arg0) throws Exception {
				File f = new File("tmp_image");
				f.mkdir();
				System.out.println("tmp_dir: " + f.getAbsolutePath());
			}
		});
                
        final Accumulator<Integer> totalCount = sc.accumulator(0);
        
        train_data.foreach(new VoidFunction<Record>() {
			transient int count;
			transient LMDBWriter dbWriter;
			
			private void readObject(ObjectInputStream in) throws ClassNotFoundException, IOException {
				in.defaultReadObject();
				dbWriter = new LMDBWriter("tmp_image", 1000);
				count = 0;
			}
			
			@Override
			public void call(Record arg0) throws Exception {
				dbWriter.putSample(arg0);
				
				if((++count % 1000) == 0) {
					System.out.println(String.format("%d images saved...", count));
				}
				
				totalCount.add(1);
			}
			
			@Override
			public void finalize() {
				try {
					dbWriter.closeLMDB();
				} catch(Exception e) {
					e.printStackTrace();
				}
				
				System.out.println(String.format("Total %d images saved...", count));
			}
		});
        
        System.out.println(String.format("%d data records found...", totalCount.value()));
        
        String serverHost = InetAddress.getLocalHost().getHostAddress();
        final int port = 10020;
        System.out.println("ParameterServer host: " + serverHost);

        DistNet net = new DistNet(args[0], args[1]);
        DistAsyncEASGDSolver driver = new DistAsyncEASGDSolver(net, serverHost, port, param);
                
        System.out.println("Start Learning...");
        Date startTime = new Date();
        
        
        driver.trainWithLMDB(rddkey, "tmp_image", numExecutors);
        driver.saveWeight("final_weight.bin");
        
        sc.close();
        
        float[] ret = driver.test(Arrays.asList(test_sample));
        
        Date endTime = new Date();
		long time = endTime .getTime() - startTime.getTime();
        System.out.println(String.format("Training time: %f secs", (double) time / 1000));
        System.out.println(String.format("Accuracy: %f ", ret[0]));
        
    }
}
