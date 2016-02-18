package org.acl.deepspark.utils;

import org.apache.hadoop.io.ArrayPrimitiveWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;

import scala.Tuple2;

public class LabelPrinter {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SparkConf conf = new SparkConf().setAppName("ImagenetSampler")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        
        JavaSparkContext sc = new JavaSparkContext(conf);
        
        int numExecutors = conf.getInt("spark.executor.instances", -1);
        System.out.println("number of executors = " + numExecutors);
        System.out.println("Data Loading...");
        JavaPairRDD<FloatWritable, ArrayPrimitiveWritable> train_seq = 
        		sc.sequenceFile("imagenet_sampled.hsf", FloatWritable.class, ArrayPrimitiveWritable.class);
        
        train_seq.foreach(new VoidFunction<Tuple2<FloatWritable,ArrayPrimitiveWritable>>() {
			
			@Override
			public void call(Tuple2<FloatWritable, ArrayPrimitiveWritable> arg0) throws Exception {
				System.out.println(arg0._1.get() + " " + ((float[]) arg0._2.get()).length);
			}
		});
        sc.close();
	}

}
