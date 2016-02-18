package org.acl.deepspark.nn.driver;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.acl.deepspark.data.Record;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.DistNet;
import org.acl.deepspark.nn.async.ParameterClient;
import org.acl.deepspark.nn.async.ParameterServer;
import org.acl.deepspark.utils.WeightUtil;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.VoidFunction;

import caffe.Caffe.SolverParameter.SolverMode;


/**
 * Created by Jaehong on 2015-07-31.
 */
public class DistAsyncNeuralNetRunner implements Serializable {

    /**
	 * 
	 */
	private static final long serialVersionUID = -5070368428661536358L;

	private transient DistNet net;

    private String host;
    private int[] port;
    private int innerLoopSize; 

    public DistAsyncNeuralNetRunner(DistNet net, String host, int[] port, int k) {
        this.net = net;
        this.host = host;
        this.port = port;
        this.innerLoopSize = k;
    }

    public void train(JavaRDD<Record> data) throws IOException{
        final int dataSize = (int) data.cache().count();
        net.setCPU();
        ParameterServer server = new ParameterServer(net, port[0], port[1]);
        server.startServer();
        
        final byte[] solverParams = net.getSolverSetting().toByteArray();
        final byte[] netParams = net.getNetConf().toByteArray();
        
        System.out.println("Training start...");
        data.foreachPartition(new VoidFunction<Iterator<Record>>() {
            private static final long serialVersionUID = -4641037124928675165L;

			public void call(Iterator<Record> samples) throws Exception {
            	List<Record> sampleList = new ArrayList<Record>();
                while (samples.hasNext())
                    sampleList.add(samples.next());
                
                System.out.println("Local data size: " + sampleList.size());
                // init net
                DistNet net = new DistNet(solverParams, netParams);
                SolverMode mode = net.getSolverSetting().getSolverMode();
                if(mode == SolverMode.GPU) {
                	System.out.println("running on GPU...");
                	net.setGPU();
                } else {
                	System.out.println("running on CPU...");
                	net.setCPU();
                }
                
                int settingSize = sampleList.size();
                int iterLimit = sampleList.size() / net.getBatchSize();
                settingSize -= sampleList.size() % net.getBatchSize();
                
                // get initial weight
                net.setWeights(ParameterClient.getWeights(host, port[1]));
                List<Weight> cumulativeGrads = net.getGradients();
                
                int listSize = sampleList.size();
                int iteration = net.getIteration();
                int localIter = iteration * listSize / dataSize / innerLoopSize;
                
                System.out.println("iteration : " + localIter * innerLoopSize);
                
                int counter = 0;
                for (int i = 0; i < localIter; i++) {
                	WeightUtil.clear(cumulativeGrads);
                	for(int k =0; k < innerLoopSize; k++) {
	                	if(counter % iterLimit == 0) {  // iteration complete
	                		Collections.shuffle(sampleList);
	                		List<Record> subSample = sampleList.subList(0, settingSize);
	                		net.setTrainData(net.buildBlobs(subSample));	                		
	                	}	
	                		
	                	System.out.println(String.format("%d th iteration", ++counter));
	                    net.train();
	                    WeightUtil.addFromTo(net.getGradients(), cumulativeGrads);
                	}
                	ParameterClient.sendDelta(host, port[0], cumulativeGrads);
                	net.setWeights(ParameterClient.getWeights(host, port[1]));
                }
            }
        });

        System.out.println("Training complete...");
        server.stopServer();
        net.setGPU();
    }


    public float[] test(List<Record> data) {
    	net.setTestData(net.buildBlobs(data));
        return net.test();
    }
}
