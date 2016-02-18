package org.acl.deepspark.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;

import org.acl.deepspark.data.Record;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

public class MnistLoader implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 4845357475294611873L;
	private static final int dimRows = 28;
	
	public static Record[] loadIntoSamples(String path, boolean normalize) {
		System.out.println("Data Loading...");
		float label;
		BufferedReader reader = null;
		ArrayList<Record> samples = new ArrayList<Record>();
		
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = null;
			String[] feature = null;
			while((line = reader.readLine()) != null) {
				feature = line.split("\t");
				label = Float.parseFloat(feature[dimRows * dimRows]);
				
				float[] featureVec = new float[dimRows * dimRows];
				
				for(int i = 0; i < feature.length - 1;i++)
					featureVec[i] = Float.parseFloat(feature[i]);
				
				Record s = new Record();
				s.data = featureVec;
				if (normalize) {
					for(int i = 0; i < s.data.length; i++)
						s.data[i] /= 255;
				}
					
				s.label = label;
				samples.add(s);
			}
			
		} catch(IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch(IOException e) {}
			}
		}
		
		Record[] arr = new Record[samples.size()];
		arr = samples.toArray(arr);

		System.out.println(String.format("Loaded %d samples from %s", samples.size(), path));
		return arr;
	}
	
	public static JavaRDD<Record> loadRDDFromHDFS(String path, final boolean normalize, JavaSparkContext sc) {
		JavaRDD<String> lines = sc.textFile(path);
		JavaRDD<Record> ret = lines.map(new Function<String, Record>() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			public Record call(String v1) throws Exception {
				String[] feature = v1.split("\t");
				float label = Float.parseFloat(feature[dimRows * dimRows]);
				float[] featureVec = new float[dimRows * dimRows];

				for (int i = 0; i < feature.length -1; i++)
					featureVec[i] = Float.parseFloat(feature[i]);
				
				Record s = new Record();
				
				s.data = featureVec;
				if (normalize) {
					for(int i = 0; i < s.data.length; i++)
						s.data[i] /= 255;
				}
				s.label = label;
				s.dim = new int[]{1, dimRows, dimRows};
				return s;
			}
		});
		return ret;
	}

	public static Record[] loadFromHDFS(String path, boolean normalize) {
		System.out.println("Data Loading...");
		float label;
		BufferedReader reader = null;
		ArrayList<Record> samples = new ArrayList<Record>();
		
		try {
			Path p = new Path(path);
			FileSystem fs = FileSystem.get(new Configuration());
			reader = new BufferedReader(new InputStreamReader(fs.open(p)));
			
			String line = null;
			String[] feature = null;

			while ((line = reader.readLine()) != null) {
				feature = line.split("\t");
				label = Float.parseFloat(feature[dimRows * dimRows]);
				float[] featureVec = new float[dimRows * dimRows];
				for (int i = 0; i < feature.length -1; i++)
					featureVec[i] = Float.parseFloat(feature[i]);

				Record s = new Record();
				s.data = featureVec;
				if (normalize) {
					for(int i = 0; i < s.data.length; i++)
						s.data[i] /= 255;
				}
				s.label = label;
				s.dim = new int[]{1, dimRows, dimRows};
				samples.add(s);
			}

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {}
			}
		}

		Record[] arr = new Record[samples.size()];
		arr = samples.toArray(arr);

		System.out.println(String.format("Loaded %d samples from %s", samples.size(), path));
		return arr;
	}
}
