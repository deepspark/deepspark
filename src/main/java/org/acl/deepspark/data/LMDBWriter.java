package org.acl.deepspark.data;

import java.io.Serializable;

import org.fusesource.lmdbjni.Constants;
import org.fusesource.lmdbjni.Cursor;
import org.fusesource.lmdbjni.Database;
import org.fusesource.lmdbjni.Entry;
import org.fusesource.lmdbjni.Env;
import org.fusesource.lmdbjni.GetOp;
import org.fusesource.lmdbjni.Transaction;

import com.google.protobuf.ByteString;

import caffe.Caffe.Datum;

public class LMDBWriter implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3966228856774805437L;
	private Env env;
	private Database dbInstance;
	
	private Transaction tx;
	private int th;
	private int count, totalCount;
	
	public LMDBWriter(String path) {
		this(path,1000);		
	}
	
	public LMDBWriter(String path, int th) {
		
		boolean isOpen = false;
		try {
			env = new Env();
			env.setMapSize(1099511627776L);
			env.open(path, Constants.NOLOCK);
			
			dbInstance = env.openDatabase();
			this.th = th;
			isOpen = true;
			
			count = 0;
			totalCount = 0;
		} catch(Exception e) {
			e.printStackTrace();
		}
		finally {
			if (isOpen) {
				System.out.println("lmdb opened successfully");
			}
		}
	}
	
	
	private Datum convert2Datum(Record sample) {	
		byte[] bytes = new byte[sample.data.length];
		for (int i = 0 ; i< bytes.length; i++)
			bytes[i] = (byte) ((int) sample.data[i] & 0xff);
		
		Datum datum = Datum.newBuilder().setWidth(sample.dim[2])
										.setHeight(sample.dim[1])
										.setChannels(sample.dim[0])
										.setData(ByteString.copyFrom(bytes))
										.setLabel((int) sample.label)
										.build();
		return datum;
	}
	
	public void putSample(Record sample) {
		if(count == th) {
			tx.commit();
			count = 0;
		}
		
		if(count == 0)
			tx = env.createWriteTransaction();
		
		Datum d = convert2Datum(sample);
		
		byte[] array =d.toByteArray();
		
		dbInstance.put(tx,String.format("%08d",totalCount++).getBytes(), array);
		count++;
	}
	
	public void closeLMDB() {
		if(count !=0)
			tx.commit();
		/*
		if (dbInstance != null && dbInstance.isAllocated())
			dbInstance.close();
		if (env != null && env.isAllocated())
			env.close();
		*/
		System.out.println("DB successfully closed.");
	}
	
	public void printAll() {
		Transaction tx = env.createReadTransaction();
		try {
			Cursor cursor = dbInstance.openCursor(tx);
			try {
				for (Entry entry = cursor.get(GetOp.FIRST); entry != null; entry = cursor.get(GetOp.NEXT)) {
					String key = new String(entry.getKey());
					String value = new String(entry.getValue());
					//System.out.println(String.format("key:%s, val:%s", key, value));
					System.out.println(String.format("key:%s", key));
				} 
			} finally {
				cursor.close();
			}
		} finally {
			tx.commit();
		}
	}
	
	@Override
	public void finalize() {
		closeLMDB();
	}
}
