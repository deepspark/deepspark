package org.acl.deepspark.caffe;

import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.nio.Buffer;
import java.util.ArrayList;
import java.util.List;

import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.Message;
import com.google.protobuf.TextFormat;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import caffe.Caffe.NetParameter;
import caffe.Caffe.SolverParameter;

public class CaffeWrapper implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 2977299799611018488L;
	private transient Pointer ptrSolver;
	private transient SolverParameter solverSpec;
	private transient NetParameter netSpec;
	private transient Buffer train_data, train_label, test_data, test_label;
	
	public static final int CAFFE_CPU = 0;
	public static final int CAFFE_GPU = 1;
	
	private enum Phase {
		TRAINING, TEST
	}
	
	private interface CaffeInterface extends Library {
		public CaffeInterface INSTANCE = (CaffeInterface) Native.loadLibrary("caffeext", CaffeInterface.class);
		
		public int testCuda();
		
		public Pointer allocateSolver(byte[] solverspec, int solverspeclength, byte[] netspec, int netspeclength);
		public void releaseSolver(Pointer ptr);
		public void setTrainData(Pointer ptr, String data_layer_name, float[] data, float[] label, int num);
		public void setTrainBuffer(Pointer ptr, String data_layer_name, Pointer data, Pointer label, int num);
		public void setTestData(Pointer ptr, int netIdx, String data_layer_name, float[] data, float[] label, int num);
		public void setTestBuffer(Pointer ptr, int netIdx, String data_layer_name, Pointer data, Pointer label, int num);
		
		public int getNumWeight(Pointer solverPtr, int pid);
		public int getNumLearnableLayer(Pointer solverPtr);
		
		public void getGradient(Pointer solverPtr, float[] params, int offset, int num, int pid);
		public void setGradient(Pointer solverPtr, float[] params, int offset, int num, int pid);
		public void getWeight(Pointer solverPtr, float[] params, int offset, int num, int pid);
		public void setWeight(Pointer solverPtr, float[] params, int offset, int num, int pid);
		
		public float train(Pointer solver);
		public void update(Pointer solver);
		public Pointer test(Pointer solver);
		public void destroy(Pointer ptr);
		
		public void set_mode(int data);
		public void snapshotNet(Pointer solver,String filename);
	}
	
	public void snapshot(String filename) {
		CaffeInterface.INSTANCE.snapshotNet(ptrSolver, filename);
	}
	
	public CaffeWrapper(String solverName, String netName)  {
		System.out.println("loading solver from :" + solverName);
		System.out.println("loading net from :" + netName);
		try {
			byte[] solverParams = parseCaffeProto(solverName, SolverParameter.newBuilder());
			byte[] netParams = parseCaffeProto(netName, NetParameter.newBuilder());
			solverSpec = SolverParameter.parseFrom(solverParams);
			netSpec = NetParameter.parseFrom(netParams);
			
			ptrSolver = CaffeInterface.INSTANCE.allocateSolver(solverParams, solverParams.length, netParams, netParams.length);
		} catch(IOException e) {
			e.printStackTrace();
			ptrSolver = Pointer.NULL;
		}
	}
	
	public CaffeWrapper(byte[] solverParams, byte[] netParams) {
		try {
			solverSpec = SolverParameter.parseFrom(solverParams);
			netSpec = NetParameter.parseFrom(netParams);
			ptrSolver = CaffeInterface.INSTANCE.allocateSolver(solverParams, solverParams.length, netParams, netParams.length);
		} catch (InvalidProtocolBufferException e) {
			e.printStackTrace();
		}
	}
	
	public SolverParameter getSolverParameter() {
		return solverSpec;
	}
	
	public NetParameter getNetParameter() {
		return netSpec;
	}
	
	public void clearSolver() {
		checkValidity();
		CaffeInterface.INSTANCE.releaseSolver(ptrSolver);
		ptrSolver = Pointer.NULL;
	}
	
	public void setTrainData(String name, float[] data, float[] label, int num) {
		checkValidity();
		CaffeInterface.INSTANCE.setTrainData(ptrSolver, name, data, label, num);
	}
	
	public void setTrainBuffer(String name, Buffer data, Buffer label, int num) {
		checkValidity();
		train_data = data;
		train_label = label;
		CaffeInterface.INSTANCE.setTrainBuffer(ptrSolver, name, Native.getDirectBufferPointer(data),
				Native.getDirectBufferPointer(label), num);
	}
	
	public void setTestData(String name, float[] data, float[] label, int num) {
		checkValidity();
		CaffeInterface.INSTANCE.setTestData(ptrSolver, 0, name, data, label, num);
	}
	
	public void setTestBuffer(String name, Buffer data, Buffer label, int num) {
		checkValidity();
		test_data = data;
		test_label = label;
		CaffeInterface.INSTANCE.setTestBuffer(ptrSolver, 0, name, Native.getDirectBufferPointer(data),
				Native.getDirectBufferPointer(label), num);
	}
	
	public int getNumberLearnableLayer() {
		checkValidity();
		return CaffeInterface.INSTANCE.getNumLearnableLayer(ptrSolver);
	}
	
	public int getNumWeight(int pid) {
		
		checkValidity();
		return CaffeInterface.INSTANCE.getNumWeight(ptrSolver, pid);
	}
	
	public List<float[]> getGradients() {
		List<float[]> ret = new ArrayList<float[]>();
		int numLayer = getNumberLearnableLayer();
		for(int i =0; i < numLayer; i++) {
			int size = getNumWeight(i);
			ret.add(getGradient(0,size, i));	
		}
		
		return ret;
	}
	
	public List<float[]> getWeights() {
		List<float[]> ret = new ArrayList<float[]>();
		for(int i =0; i < getNumberLearnableLayer(); i++) {
			int size = getNumWeight(i);
			ret.add(getWeight(0,size,i));
		}
		
		return ret;
	}
	
	public void setWeights(List<float[]> params) {
		for(int i =0; i < params.size(); i++)
			setWeight(params.get(i), 0, i);
	}
	
	public void setGradients(List<float[]> params) {
		for(int i =0; i < params.size(); i++)
			setGradient(params.get(i), 0, i);
	}
	
	public float[] getWeight(int offset, int num, int pid) {
		checkValidity();
		float[] arr = new float[num];
		CaffeInterface.INSTANCE.getWeight(ptrSolver, arr, offset, num, pid);
		return arr;
	}

	public float[] getGradient(int offset, int num, int pid) {
		checkValidity();
		float[] arr = new float[num];
		CaffeInterface.INSTANCE.getGradient(ptrSolver, arr, offset, num, pid);
		return arr;
	}
	
	public void setGradient(float[] params, int offset, int pid) {
		checkValidity();
		CaffeInterface.INSTANCE.setGradient(ptrSolver, params, offset, params.length, pid);
	}
	
	public void setWeight(float[] params, int offset, int pid) {
		checkValidity();
		CaffeInterface.INSTANCE.setWeight(ptrSolver, params, offset, params.length, pid);
	}
	
	public float train() {
		checkValidity();
		checkDataReady(Phase.TRAINING);
		
		return CaffeInterface.INSTANCE.train(ptrSolver);
	}
	
	public float trainWithLocal() {
		checkValidity();
		return CaffeInterface.INSTANCE.train(ptrSolver);
	}
	
	private void checkDataReady(Phase phase) throws RuntimeException {
		switch(phase) {
		case TRAINING:
			if(train_data == null)
				throw new RuntimeException("Training data have not assinged..");
			if(train_label == null)
				throw new RuntimeException("Training label have not assinged..");
			break;
		case TEST:
			if(test_data == null)
				throw new RuntimeException("Test data have not assinged..");
			if(test_label == null)
				throw new RuntimeException("Test label have not assinged..");
			break;
		}
		return;
	}

	public float[] test() {
		checkValidity();
		checkDataReady(Phase.TEST);
		
		Pointer p = CaffeInterface.INSTANCE.test(ptrSolver);
		int size = p.getInt(0);
		float[] ret = p.getFloatArray(4 , size);
		CaffeInterface.INSTANCE.destroy(p);
		
		return ret;
	}
	
	public float[] testWithLMDB() {
		checkValidity();
		
		Pointer p = CaffeInterface.INSTANCE.test(ptrSolver);
		int size = p.getInt(0);
		float[] ret = p.getFloatArray(4 , size);
		CaffeInterface.INSTANCE.destroy(p);
		
		return ret;
	}
	
	public void set_mode(int mode) {
		switch (mode) {
		case CAFFE_CPU:
			CaffeInterface.INSTANCE.set_mode(mode);
			break;
		case CAFFE_GPU:
			if(CaffeInterface.INSTANCE.testCuda() > 0)
				CaffeInterface.INSTANCE.set_mode(mode);
			else {
				System.out.println("GPU Initialization failed..");
				CaffeInterface.INSTANCE.set_mode(CAFFE_CPU);
			}
			break;
		default:
			break;
		}
	}
	
	public void manualUpdate() {
		checkValidity();
		CaffeInterface.INSTANCE.update(ptrSolver);
	}
	
	private void checkValidity() throws RuntimeException {
		if(ptrSolver.equals(Pointer.NULL))
			throw new RuntimeException("Invalid solver pointer..");
	}
	
	private byte[] parseCaffeProto(String filename, Message.Builder builder) throws IOException {
		FileReader reader = new FileReader(filename);
		TextFormat.merge(reader, builder);
		return builder.build().toByteArray();
	}
}
