package org.acl.deepspark.utils;

import java.util.Random;

import org.acl.deepspark.data.Record;
import org.apache.hadoop.io.ArrayPrimitiveWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

public class ImagenetSampler {

	public static void main(String[] args) {
    	SparkConf conf = new SparkConf().setAppName("ImagenetSampler")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        
        JavaSparkContext sc = new JavaSparkContext(conf);
        
        final int number = Integer.parseInt(args[0]);
        
        int numExecutors = conf.getInt("spark.executor.instances", -1);
        System.out.println("number of executors = " + numExecutors);
        System.out.println("Data Loading...");
        JavaPairRDD<FloatWritable, ArrayPrimitiveWritable> train_seq = 
        		sc.sequenceFile(args[1], FloatWritable.class, ArrayPrimitiveWritable.class);
        
        JavaRDD<Record> train_samples = train_seq.map(new Function<Tuple2<FloatWritable,ArrayPrimitiveWritable>, Record>() {
        	@Override
			public Record call(Tuple2<FloatWritable, ArrayPrimitiveWritable> arg0) throws Exception {
				Record d = new Record();
				d.label = arg0._1.get();
				d.data = (float[]) arg0._2.get();
				d.dim = new int[]{3,256,256};
				return d;
			}
		}).filter(new Function<Record, Boolean>() {
			
			@Override
			public Boolean call(Record arg0) throws Exception {
				if(arg0.label < number)
					return true;
				else
					return false;
			}
		});
        
        // shuffle data
        JavaPairRDD<Integer, Record> shuffled = train_samples.mapToPair(new PairFunction<Record, Integer, Record>() {
        	Random r = new Random();
			@Override
			public Tuple2<Integer, Record> call(Record arg0) throws Exception {
				return new Tuple2<Integer, Record>(r.nextInt(), arg0);
			}
		});
        
        JavaRDD<Record> shuffled_data = shuffled.sortByKey().values();
        
        JavaPairRDD<FloatWritable, ArrayPrimitiveWritable> toSave = shuffled_data.mapToPair(new PairFunction<Record, FloatWritable, ArrayPrimitiveWritable>() {

			@Override
			public Tuple2<FloatWritable, ArrayPrimitiveWritable> call(Record arg0) throws Exception {
				System.out.println(arg0.label);
				Tuple2<FloatWritable,ArrayPrimitiveWritable> result = new Tuple2<FloatWritable, ArrayPrimitiveWritable>(new FloatWritable(arg0.label),
						new ArrayPrimitiveWritable(arg0.data));
				return result;
			}
		});
        
        toSave.saveAsHadoopFile(args[2], FloatWritable.class, ArrayPrimitiveWritable.class, SequenceFileOutputFormat.class);
        
        sc.close();	
	}
}
