package org.acl.deepspark.nn.driver;

import java.net.InetAddress;
import java.util.Arrays;
import java.util.Date;

import org.acl.deepspark.data.ByteRecord;
import org.acl.deepspark.nn.DistNet;
import org.acl.deepspark.nn.async.easgd.DistAsyncEASGDSolver;
import org.acl.deepspark.utils.DeepSparkConf.DeepSparkParam;
import org.acl.deepspark.utils.DeepSparkParamLoader;
import org.apache.hadoop.io.ArrayPrimitiveWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import scala.Tuple2;

/**
 * Created by Hanjoo on 2016-01-18.
 */
public class AsyncEASGDImageNetVarTest {
    public static void main(String[] args) throws Exception {
    	SparkConf conf = new SparkConf().setAppName("AsyncImageNetTest")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        
        JavaSparkContext sc = new JavaSparkContext(conf);
        
        DistNet net = new DistNet(args[0], args[1], new int[]{3, 256, 256});
        DeepSparkParam param = DeepSparkParamLoader.readConf(args[2]);
        
        String serverHost = InetAddress.getLocalHost().getHostAddress();
        final int port = 10020;
        System.out.println("ParameterServer host: " + serverHost);
        DistAsyncEASGDSolver driver = new DistAsyncEASGDSolver(net, serverHost, port, param);
        int numExecutors = conf.getInt("spark.executor.instances", -1);
        System.out.println("number of executors = " + numExecutors);
        
        System.out.println("Data Loading...");
        JavaPairRDD<FloatWritable, ArrayPrimitiveWritable> train_seq = 
        		sc.sequenceFile("/imagenet_train.hsf", FloatWritable.class, ArrayPrimitiveWritable.class);

        Integer[] tempkey = new Integer[numExecutors];
        for(int i = 0; i < tempkey.length; i++)
        	tempkey[i] = i;
        
        JavaRDD<Integer> rddkey = sc.parallelize(Arrays.asList(tempkey), numExecutors);
        
        JavaRDD<ByteRecord> train_samples = train_seq.map(new Function<Tuple2<FloatWritable,ArrayPrimitiveWritable>, ByteRecord>() {
        	@Override
			public ByteRecord call(Tuple2<FloatWritable, ArrayPrimitiveWritable> arg0) throws Exception {
        		ByteRecord d = new ByteRecord();
				d.label = arg0._1.get();
				d.data = (byte[]) arg0._2.get();
				d.dim = new int[]{3,256,256};
				return d;
			}
		});
        
                // spill to local
        driver.prepareLocalByte(train_samples, "tmp_data", numExecutors);
        
        train_samples.unpersist();
        System.out.println("Start Learning...");
        Date startTime = new Date();
        
        //train local
        driver.trainWithLMDBVarTau(rddkey, "tmp_image", numExecutors);
        
        Date endTime = new Date();
        	
        long time = endTime.getTime() - startTime.getTime();
        System.out.println(String.format("Training time: %f secs", (double) time / 1000));
        
        sc.close();
    }
}
