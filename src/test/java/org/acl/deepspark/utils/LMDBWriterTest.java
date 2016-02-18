package org.acl.deepspark.utils;

import org.acl.deepspark.data.LMDBWriter;
import org.acl.deepspark.data.Record;

import caffe.Caffe.Datum;

public class LMDBWriterTest {
	public static void main(String[] args) {
		//Sample[] samples = CIFARLoader.loadIntoSamples("/home/acl/CIFAR-10/train_batch.bin", false);

		LMDBWriter dbWriter = new LMDBWriter("/home/acl/data_sample");
		
		/*
		for (int i = 0 ; i < samples.length; i++) {
			dbWriter.putSample(samples[i]);		
		}
				
		*/
		dbWriter.printAll();
		dbWriter.closeLMDB();
	}
		
}
