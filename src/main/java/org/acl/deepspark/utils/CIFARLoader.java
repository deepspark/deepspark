package org.acl.deepspark.utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;

import org.acl.deepspark.data.Record;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class CIFARLoader implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 2453031569679527445L;
		
	private static final int dimRows = 32;
	private static final int dimCols = 32;
	private static final int channel = 3;
			
	public static Record[] loadIntoSamples(String path, boolean normalize) {
		int label;
		byte[] data = new byte[channel * dimRows * dimCols];
		ArrayList<Record> samples = new ArrayList<Record>();

		FileInputStream in = null;
		try {
			in = new FileInputStream(path);
			while ((label = in.read()) != -1) {
				float[] featureVec = new float [channel * dimRows * dimCols];
				
				int value;
				int length = featureVec.length;

				in.read(data);
				float mean = 0;
				for (int i = 0 ; i < length; i++) {
					value = (int) data[i]&0xff;
					featureVec[i] = (float) value;
					mean += value;
				}
				
				mean /= length;

				Record s = new Record();
				s.data = featureVec;
				if (normalize) {
					for(int i =0; i < length; i++ ) {
						s.data[i] -= mean;
					}
				}
				s.label = label;
				s.dim = new int[]{channel, dimRows, dimCols};
				samples.add(s);
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (in != null) {
				try {
					in.close();
				} catch (IOException e) {}
			}
		}

		Record[] arr = new Record[samples.size()];
		arr = samples.toArray(arr);

		System.out.println(String.format("Loaded %d samples from %s", samples.size(), path));
		return arr;
	}

	public static Record[] loadFromHDFS(String path, boolean normalize) {
		int label;
		byte[] data = new byte[channel * dimRows * dimCols];
		ArrayList<Record> samples = new ArrayList<Record>();

		FSDataInputStream in = null;
		try {
			Path p = new Path(path);
			FileSystem fs = FileSystem.get(new Configuration());
			in = fs.open(p);

			while ((label = in.read()) != -1) {
				float[] featureVec = new float [channel * dimRows * dimCols];
				
				int value;
				int length = featureVec.length;

				in.read(data);
				
				float mean = 0;
				for (int i = 0 ; i < length; i++) {
					value = (int) data[i]&0xff;
					featureVec[i] = (float) value;
					mean += value;
				}
				
				mean /= length;

				Record s = new Record();
				s.data = featureVec;
				if (normalize) {
					for(int i =0; i < length; i++ ) {
						s.data[i] -= mean;
					}
				}
				s.label = label;
				s.dim = new int[]{channel, dimRows, dimCols};
				samples.add(s);
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (in != null) {
				try {
					in.close();
				} catch (IOException e) {}
			}
		}

		Record[] arr = new Record[samples.size()];
		arr = samples.toArray(arr);

		System.out.println(String.format("Loaded %d samples from %s", samples.size(), path));
		return arr;
	}

	public static Record[] loadFromHDFSText(String path, boolean normalize) {
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
				label = Float.parseFloat(feature[channel*dimRows*dimRows]);
				
				float[] featureVec = new float[channel*dimRows*dimCols];
				
				float mean = 0;
				for (int i = 0; i < feature.length -1; i++) {
					featureVec[i] = Float.parseFloat(feature[i]);
					mean += featureVec[i];
				}
				
				mean /= feature.length;

				Record s = new Record();
				s.data = featureVec;
				if (normalize) {
					for(int i =0; i < feature.length -1; i++ ) {
						s.data[i] -= mean;
					}
				}
				s.label = label;
				s.dim = new int[]{channel, dimRows, dimCols};
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