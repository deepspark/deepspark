package org.acl.deepspark.nn;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.acl.deepspark.caffe.CaffeWrapper;
import org.acl.deepspark.data.ByteRecord;
import org.acl.deepspark.data.LMDBWriter;
import org.acl.deepspark.data.Record;
import org.acl.deepspark.data.RecordBatch;
import org.acl.deepspark.data.Weight;

import com.google.protobuf.TextFormat;

import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.NetStateRule;
import caffe.Caffe.Phase;
import caffe.Caffe.SolverParameter;

public class DistNet implements Serializable {
    /**
	 * 
	 */
	private static final long serialVersionUID = -4406616544404720649L;
	private CaffeWrapper netLearner;
    private String trainingDataName;
    private String testDataName;
    private int[] inputDim;
    
    public DistNet(String solverSpec, String netSpec, int[] dummySize) {
    	makeDummy(netSpec,dummySize);
    	this.netLearner = new CaffeWrapper(solverSpec, netSpec);
        initialize();
    }
    
    public DistNet(String solverSpec, String netSpec) {
    	this.netLearner = new CaffeWrapper(solverSpec, netSpec);
        initialize();
    }

    private void makeDummy(String netSpec, int[] dummySize) {
    	// make missed dirs
    	FileReader reader;
		try {
			reader = new FileReader(netSpec);
			NetParameter.Builder netBuilder = NetParameter.newBuilder();
			TextFormat.merge(reader, netBuilder);
			NetParameter netparam = netBuilder.build();
			
			List<LayerParameter> list = netparam.getLayerList();
			
			
			// for training layer
			int idx = findDataLayer(list, Phase.TRAIN);			
			LayerParameter l = list.get(idx);
			if(l.getType().equals("Data")) {
				int batchsize = l.getDataParam().getBatchSize();
				String source = l.getDataParam().getSource();
				File dir = new File(source);
				if(!dir.exists())
					dir.mkdirs();
				
				LMDBWriter d = new LMDBWriter(source, batchsize);
				ByteRecord s = new ByteRecord();
				s.data = new byte[dummySize[0] * dummySize[1] * dummySize[2]];
				s.label = 0;
				s.dim = new int[] {dummySize[0], dummySize[1], dummySize[2]};
						
				for(int i = 0; i < batchsize; i++) {
					d.putSample(s);
				}
				d.closeLMDB();
			}
			
			// for testing layer
			idx = findDataLayer(list, Phase.TEST);			
			l = list.get(idx);
			if(l.getType().equals("Data")) {
				int batchsize = l.getDataParam().getBatchSize();
				String source = l.getDataParam().getSource();
				File dir = new File(source);
				if(!dir.exists())
					dir.mkdirs();
				
				LMDBWriter d = new LMDBWriter(source, batchsize);
				Record s = new Record();
				s.data = new float[dummySize[0] * dummySize[1] * dummySize[2]];
				s.label = 0;
				s.dim = new int[] {dummySize[0], dummySize[1], dummySize[2]};
				
				for(int i = 0; i < batchsize; i++) {
					d.putSample(s);
				}
				
				d.closeLMDB();
			}
			
			
		} catch (IOException e1) {
			e1.printStackTrace();
		}
    }
    
    public DistNet(byte[] solverSpec, byte[] netSpec) {
    	this.netLearner = new CaffeWrapper(solverSpec, netSpec);
    	initialize();
    }
    
    public int getTrainingBatchSize() {
    	List<LayerParameter> layers = netLearner.getNetParameter().getLayerList();
        int trainIdx = findDataLayer(layers, Phase.TRAIN);
        LayerParameter trainingDataLayer = layers.get(trainIdx);
        return trainingDataLayer.getMemoryDataParam().getBatchSize();
    }
    
    private void initialize() {
    	List<LayerParameter> layers = netLearner.getNetParameter().getLayerList();
        int trainIdx = findDataLayer(layers, Phase.TRAIN);
        
        if(trainIdx == -1) {
        	// error handling
        	System.out.println("cannot find training data layer...");
        } else {
	        LayerParameter trainingDataLayer = layers.get(trainIdx);
	        trainingDataName = trainingDataLayer.getName();
	        
	        String layerType = trainingDataLayer.getType();
	        if(layerType.equals("MemoryData")) {
		        inputDim = new int[4]; // batchSize, channels, height, width;
		        inputDim[0] = trainingDataLayer.getMemoryDataParam().getBatchSize();// batchSize;
		        inputDim[1] = trainingDataLayer.getMemoryDataParam().getChannels();// channels;
		        inputDim[2] = trainingDataLayer.getMemoryDataParam().getHeight();// height;
		        inputDim[3] = trainingDataLayer.getMemoryDataParam().getWidth();// width;
		        
		        System.out.println("Iteration : " + netLearner.getSolverParameter().getMaxIter());
		        System.out.println("Learning Rate : " + netLearner.getSolverParameter().getBaseLr());
		        System.out.println(String.format("batchSize: %d", inputDim[0]));
		        System.out.println(String.format("channel: %d", inputDim[1]));
		        System.out.println(String.format("height: %d", inputDim[2]));
		        System.out.println(String.format("width: %d", inputDim[3]));
		        System.out.println(String.format("momentum: %4f", netLearner.getSolverParameter().getMomentum()));
		        System.out.println(String.format("decayLambda: %4f", netLearner.getSolverParameter().getWeightDecay()));
	        } else if(layerType.equals("Data")) {
	        	inputDim = new int[1];
	        	inputDim[0] = trainingDataLayer.getDataParam().getBatchSize();
	        	
	        	System.out.println("Iteration : " + netLearner.getSolverParameter().getMaxIter());
		        System.out.println("Learning Rate : " + netLearner.getSolverParameter().getBaseLr());
		        System.out.println(String.format("batchSize: %d", inputDim[0]));
		        System.out.println(String.format("momentum: %4f", netLearner.getSolverParameter().getMomentum()));
		        System.out.println(String.format("decayLambda: %4f", netLearner.getSolverParameter().getWeightDecay()));
		        System.out.println(String.format("Training Source: %s", trainingDataLayer.getDataParam().getSource()));
	        }
        }
        
        int testIdx = findDataLayer(layers, Phase.TEST);	
        if(testIdx == -1 ) {
        	// error handling
        } else {
	        
	        LayerParameter testingDataLayer = layers.get(testIdx);
	        testDataName = testingDataLayer.getName();
        }
    }

    private int findDataLayer(List<LayerParameter> list, Phase phase) {
    	int res = -1;
    	for(int i = 0; i < list.size();i++) {
    		LayerParameter l = list.get(i);
    		List<NetStateRule> rule = l.getIncludeList();
    		for(int j =0; j < rule.size(); j++)
	    		if( l.getType().contains("Data") &&
	    				rule.get(j).hasPhase() &&
	    				rule.get(j).getPhase() == phase) {
	    			res = i;
	    			return res;
	    		}
    	}
    	
    	return res;
    }
    
    public int findDataLayer(Phase phase) {
    	return findDataLayer(netLearner.getNetParameter().getLayerList(), phase);
    }
    
    public int getIteration() {
    	return netLearner.getSolverParameter().getMaxIter();
    }
    
    public void setTrainData(RecordBatch dataset) {
        netLearner.setTrainBuffer(trainingDataName, dataset.data, dataset.label, dataset.size);
    }
    
    public void setTestData(RecordBatch dataset) {
        netLearner.setTestBuffer(testDataName, dataset.data, dataset.label, dataset.size);
    }
    
    public float train() {
    	return netLearner.train();
    }
    
    public float trainWithLocal() {
    	return netLearner.trainWithLocal();
    }
    
    public float[] test() {
    	return netLearner.test();
    }
    
    public float[] testWithLMDB() {
    	return netLearner.testWithLMDB();
    }
    
    public void snapshot(String filename) {
    	netLearner.snapshot(filename);
    }
    
    public void restore(String filename) {
    	netLearner.restore(filename);
    }
    
    public List<Weight> getGradients() {
    	List<float[]> list = netLearner.getGradients();
    	List<Weight> ret = new ArrayList<Weight>();
    	
    	for(int i = 0; i < list.size();i++) {
    		Weight w = new Weight();
    		w.layerIndex = i;
    		w.offset = 0;
    		w.data = list.get(i);
    		ret.add(w);
    	}
    	
    	return ret;
    }
    
    public void setGradients(List<Weight> grads) {
    	for(int i =0; i < grads.size(); i++) {
    		Weight w = grads.get(i);
    		netLearner.setGradient(w.data, w.offset, i);
    	}
    }
    
    public void manualUpdate() {
    	netLearner.manualUpdate();
    }
    
    public List<Weight> getWeights() {
    	List<float[]> list = netLearner.getWeights();
    	List<Weight> ret = new ArrayList<Weight>();
    	
    	for(int i = 0; i < list.size();i++) {
    		Weight w = new Weight();
    		w.layerIndex = i;
    		w.offset = 0;
    		w.data = list.get(i);
    		ret.add(w);
    	}
    	
    	return ret;
    }
    
    public void setWeights(List<Weight> weights) {
    	for(int i =0; i < weights.size(); i++) {
    		Weight w = weights.get(i);
    		netLearner.setWeight(w.data, w.offset, i);
    	}
    }
    
    public RecordBatch buildBlobs(List<Record> list) {
    	if(inputDim.length== 4) {
	    	RecordBatch ret = new RecordBatch(list.size(), inputDim[1], inputDim[2], inputDim[3]);
	    	
	    	for(int i = 0; i < list.size(); i++) {
	    		Record s = list.get(i);
	    		for(int j = 0; j < s.data.length; j++) {
	    			ret.data.putFloat((i* s.data.length + j) * Float.SIZE / Byte.SIZE, s.data[j]);
	    		}
	    		ret.label.putFloat(i*Float.SIZE / Byte.SIZE, s.label);
	    	}
	    	return ret;
    	} else return null;
    }
    
    public RecordBatch buildBlobs(Iterator<Record> iter) {
    	List<Record> array = new ArrayList<Record>();
    	while(iter.hasNext())
    		array.add(iter.next());
    	return buildBlobs(array);
    }
    
    public SolverParameter getSolverSetting() {
    	return netLearner.getSolverParameter();
    }
    
    public NetParameter getNetConf() {
    	return netLearner.getNetParameter();
    }

    public void setCPU() {
    	netLearner.set_mode(CaffeWrapper.CAFFE_CPU);
    }
    
    public void setGPU() {
    	netLearner.set_mode(CaffeWrapper.CAFFE_GPU);
    }
    
    public int getBatchSize() {
    	return inputDim[0];
    }
    
    public void printLearnableWeightInfo() {
    	List<Weight> list = getWeights();
    	System.out.println("<Weight Info>");
    	for(int i =0; i < list.size(); i++) {
    		Weight w = list.get(i);
    		System.out.println(String.format("Layer%d, # of parameter: %d", i, w.data.length));
    	}
    }
    
    public void close() {
    	netLearner.clearSolver();
    }
}
