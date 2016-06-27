package org.acl.deepspark.nn.async.easgd;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.acl.deepspark.data.ByteRecord;
import org.acl.deepspark.data.LMDBWriter;
import org.acl.deepspark.data.Record;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.DistNet;
import org.acl.deepspark.utils.DeepSparkConf;
import org.acl.deepspark.utils.DeepSparkConf.DeepSparkParam;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.VoidFunction;

import caffe.Caffe.SolverParameter.SolverMode;


/**
 * Created by Jaehong on 2015-07-31.
 */
public class DistAsyncEASGDSolver implements Serializable {

    /**
	 * 
	 */
	private static final long serialVersionUID = -5070368428661536358L;

	private transient DistNet net;

    private String host;
    private int port;
    
    private float movingRate;
	private int decayStep = 0;
	private float decayRate = 1;
	private int decayLimit = 0;

	private DeepSparkParam param;
	
	public DistAsyncEASGDSolver(DistNet net, String host, int port, DeepSparkConf.DeepSparkParam param) {
		this.net = net;
        this.host = host;
        this.port = port;
        this.param = param;
        
        movingRate = param.getMovingRate();
        decayStep = param.getDecayStep() / param.getPeriod();
        decayRate = param.getDecayRate();
        decayLimit = param.getDecayLimit();
    }
	
	private long startTime;
    
	private void printLog(String mesg) {
		System.out.println(String.format("[%s solver]: %s", new Timestamp(System.currentTimeMillis()),mesg));
	}
	
	private void printLogTime(String mesg) {
		System.out.println(String.format("[%f solver]: %s",  (float) (System.currentTimeMillis() - startTime ) / 1000, mesg));
	}
	
    public void train(JavaRDD<Record> data, final int numWorker) throws IOException{
    	SolverMode mode = net.getSolverSetting().getSolverMode();
        if(mode == SolverMode.GPU) {
        	printLog("running on GPU...");
        	net.setGPU();
        } else {
        	printLog("running on CPU...");
        	net.setCPU();
        }
        
    	net.printLearnableWeightInfo();
    	
        ParameterEASGDServer server = new ParameterEASGDServer(net, port, numWorker, param);
        
        server.startServer();
        
        final byte[] solverParams = net.getSolverSetting().toByteArray();
        final byte[] netParams = net.getNetConf().toByteArray();
        
        final List<Weight> initialWeight = net.getWeights();
        
        startTime = System.currentTimeMillis();
        server.setStartTime(startTime);
        printLogTime("Training start...");
        
        data.foreachPartition(new VoidFunction<Iterator<Record>>() {
            private static final long serialVersionUID = -4641037124928675165L;

			public void call(Iterator<Record> samples) throws Exception {
            	List<Record> sampleList = new ArrayList<Record>();
                while (samples.hasNext())
                    sampleList.add(samples.next());
                
                printLog("Local data size: " + sampleList.size());
                
                startTime = System.currentTimeMillis();
            	// init net
                DistNet net = new DistNet(solverParams, netParams);
                SolverMode mode = net.getSolverSetting().getSolverMode();
                if(mode == SolverMode.GPU) {
                	printLog("running on GPU...");
                	net.setGPU();
                } else {
                	printLog("running on CPU...");
                	net.setCPU();
                }
                                
                // get initial weight
                net.printLearnableWeightInfo();
                net.setWeights(initialWeight);                
                
                int settingSize = sampleList.size();
                int iterLimit = sampleList.size() / net.getBatchSize();
                settingSize -= sampleList.size() % net.getBatchSize();
                
                int iteration = net.getIteration();
                int localIter = Math.round((float)iteration / param.getPeriod());
                
                printLogTime("iteration : " + localIter * param.getPeriod());
                
                int counter = 0;
                for (int i = 0; i < localIter; i++) {
                	if(decayStep > 0 && i !=0 && decayLimit != 0 &&
                			i % decayStep == 0) {
                		decayLimit--;
                		movingRate *= decayRate; 
                		printLogTime(String.format("moving rate: %f", movingRate));
                	}
                	
                	float loss = 0;
                	long mean_train_time = 0;
                	
                	for(int k =0; k < param.getPeriod(); k++) {
                		if(counter % iterLimit == 0) {  // iteration complete
	                		Collections.shuffle(sampleList);
	                		List<Record> subSample = sampleList.subList(0, settingSize);
	                		net.setTrainData(net.buildBlobs(subSample));	                		
	                	}	
                		
	                	long start_train = System.currentTimeMillis();
	                	float loss_per_iter = net.train(); 
                		loss += loss_per_iter;
                		long end_tra = System.currentTimeMillis();
                		mean_train_time += end_tra -start_train;
                		
                		++counter;
                			                	
                	}
                	
                	printLogTime(String.format("%d th iteration, loss: %f (mean time: %d ms)", counter, loss / param.getPeriod(), mean_train_time / param.getPeriod()));
                	
                	
                	List<Weight> new_weight = ParameterEASGDClient.exchangeWeight(host, port, net.getWeights(), movingRate);
                	net.setWeights(new_weight);
                }
            }
        });

        printLogTime("Training complete...");
        server.stopServer();
        net.setGPU();
    }
    
    public void trainWithLMDB(JavaRDD<Integer> dummydata, String path, final int numWorker) throws IOException{
        SolverMode mode = net.getSolverSetting().getSolverMode();
        if(mode == SolverMode.GPU) {
        	printLog("running on GPU...");
        	net.setGPU();
        } else {
        	printLog("running on CPU...");
        	net.setCPU();
        }
        
    	net.printLearnableWeightInfo();
    	
        ParameterEASGDServer server = new ParameterEASGDServer(net, port, numWorker, param);
        
        server.startServer();
        
        final byte[] solverParams = net.getSolverSetting().toByteArray();
        final byte[] netParams = net.getNetConf().toByteArray();
        
        final List<Weight> initialWeight = net.getWeights();
        
        startTime = System.currentTimeMillis();
        server.setStartTime(startTime);
        printLogTime("Training start...");

        dummydata.foreachPartition(new VoidFunction<Iterator<Integer>>() {
            private static final long serialVersionUID = -4641037124928675165L;

			public void call(Iterator<Integer> samples) throws Exception {
				System.gc();
				
				int id = samples.next();
				printLogTime(String.format("I'm %d", id));
				
				
				startTime = System.currentTimeMillis();
            	// init net
                DistNet net = new DistNet(solverParams, netParams);
                SolverMode mode = net.getSolverSetting().getSolverMode();
                if(mode == SolverMode.GPU) {
                	printLog("running on GPU...");
                	net.setGPU();
                } else {
                	printLog("running on CPU...");
                	net.setCPU();
                }
                
               // for represent worker node snapshot
				String dirName = null; int tick = 0;
				Configuration conf=null; FileSystem fs =null;
				if(net.getSolverSetting().hasSnapshot() && id == 0) {
					conf = new Configuration();
					fs = FileSystem.get(conf);
					tick = net.getSolverSetting().getSnapshot();
					dirName = net.getSolverSetting().getSnapshotPrefix();
				}
                
                // get initial weight
                net.printLearnableWeightInfo();
                net.setWeights(initialWeight);                
                
                int iteration = net.getIteration();
                int localIter = Math.round((float)iteration / param.getPeriod());
                
                printLogTime("iteration : " + localIter * param.getPeriod());
                
                int counter = 0;
                for (int i = 0; i < localIter; i++) {
                	if(decayStep > 0 && i !=0 && decayLimit != 0 &&
                			i % decayStep == 0) {
                		decayLimit--;
                		movingRate *= decayRate; 
                		printLogTime(String.format("moving rate: %f", movingRate));
                	}
                	
                	float loss = 0;
                	long mean_train_time = 0;
                	for(int k =0; k < param.getPeriod(); k++) {
	                	long start_train = System.currentTimeMillis();
	                	float loss_per_iter = net.trainWithLocal(); 
                		loss += loss_per_iter;
                		long end_tra = System.currentTimeMillis();
                		mean_train_time += end_tra -start_train;
                		
                		++counter;
                		
	                	// save weights
	                	if(id == 0) {
		                	if(tick != 0 && counter % tick == 0) {
								String filename = String.format("%s/state_worker_iter_%d.weight", dirName, counter);
								Path path = new Path(filename);
								
								FSDataOutputStream fin = fs.create(path);
								ObjectOutputStream out = new ObjectOutputStream(fin);
								out.writeObject(net.getWeights());
								out.close();
								printLogTime(String.format("%s saved..",filename));
							}
	                	}
                	}
                	
                	printLogTime(String.format("%d th iteration, loss: %f (mean time: %d ms)", counter, loss / param.getPeriod(), mean_train_time / param.getPeriod()));
                	
                	
                	List<Weight> new_weight = ParameterEASGDClient.exchangeWeight(host, port, net.getWeights(), movingRate);
                	net.setWeights(new_weight);
                }
            }
        });

        printLogTime("Training complete...");
        server.stopServer();
        net.setGPU();
    }
    
    public void trainWithLMDBVarTau(JavaRDD<Integer> dummydata, String path, final int numWorker) throws IOException{
        SolverMode mode = net.getSolverSetting().getSolverMode();
        if(mode == SolverMode.GPU) {
        	printLog("running on GPU...");
        	net.setGPU();
        } else {
        	printLog("running on CPU...");
        	net.setCPU();
        }
        
    	net.printLearnableWeightInfo();
    	
        ParameterEASGDServer server = new ParameterEASGDServer(net, port, numWorker, param);
        
        server.startServer();
        
        final byte[] solverParams = net.getSolverSetting().toByteArray();
        final byte[] netParams = net.getNetConf().toByteArray();
        
        final List<Weight> initialWeight = net.getWeights();
        
        startTime = System.currentTimeMillis();
        server.setStartTime(startTime);
        printLogTime("Training start...");

        dummydata.foreachPartition(new VoidFunction<Iterator<Integer>>() {
            private static final long serialVersionUID = -4641037124928675165L;

			public void call(Iterator<Integer> samples) throws Exception {
				System.gc();
				
				int id = samples.next();
				printLogTime(String.format("I'm %d", id));
				
				
				startTime = System.currentTimeMillis();
            	// init net
                DistNet net = new DistNet(solverParams, netParams);
                SolverMode mode = net.getSolverSetting().getSolverMode();
                if(mode == SolverMode.GPU) {
                	printLog("running on GPU...");
                	net.setGPU();
                } else {
                	printLog("running on CPU...");
                	net.setCPU();
                }
                
               // for represent worker node snapshot
				String dirName = null; int tick = 0;
				Configuration conf=null; FileSystem fs =null;
				if(net.getSolverSetting().hasSnapshot() && id == 0) {
					conf = new Configuration();
					fs = FileSystem.get(conf);
					tick = net.getSolverSetting().getSnapshot();
					dirName = net.getSolverSetting().getSnapshotPrefix();
				}
                
                // get initial weight
                net.printLearnableWeightInfo();
                net.setWeights(initialWeight);                
                
                int iteration = net.getIteration();
                
                float lossCut = -1.0f;
                float lossCum = 0.0f;
                long mean_train_time = 0;
                int counter = 0, lastCounter = 0;
                
                for(int i =0; i < iteration; i++) {
                	if(decayStep > 0 && i !=0 && decayLimit != 0 &&
                			i % decayStep == 0) {
                		decayLimit--;
                		movingRate *= decayRate; 
                		printLogTime(String.format("moving rate: %f", movingRate));
                	}
                	
                	counter++;
                	long start_train = System.currentTimeMillis();
                	float loss_per_iter = net.trainWithLocal();
                	long end_tra = System.currentTimeMillis();
                	mean_train_time += end_tra -start_train;
                	lossCum += loss_per_iter;
            
                	// save weights
                	if(id == 0) {
	                	if(tick != 0 && counter % tick == 0) {
							String filename = String.format("%s/state_worker_iter_%d.weight", dirName, counter);
							Path path = new Path(filename);
							
							FSDataOutputStream fin = fs.create(path);
							ObjectOutputStream out = new ObjectOutputStream(fin);
							out.writeObject(net.getWeights());
							out.close();
							printLogTime(String.format("%s saved..",filename));
						}
                	}
                	
                	//initial lossCut set
                	if(counter == param.getPeriod() ) {
                		lossCut = lossCum;
                		printLogTime(String.format("%d th iteration, loss: %f (mean time: %d ms) for %d iterations", counter, lossCum / param.getPeriod(), mean_train_time / param.getPeriod(), param.getPeriod()));
                		List<Weight> new_weight = ParameterEASGDClient.exchangeWeight(host, port, net.getWeights(), movingRate);
                    	net.setWeights(new_weight);
                    	mean_train_time = 0;
                    	lossCum = 0;
                    	lastCounter = counter;
                	}
                	
                	// when lossCum meets lossCut
                	if(lossCut > 0 && lossCut < lossCum ) {
                		printLogTime(String.format("%d th iteration, loss: %f (mean time: %d ms) for %d iterations", counter, lossCum / (counter - lastCounter), mean_train_time / (counter - lastCounter),(counter - lastCounter)));
                		List<Weight> new_weight = ParameterEASGDClient.exchangeWeight(host, port, net.getWeights(), movingRate);
                    	net.setWeights(new_weight);
                    	mean_train_time = 0;
                    	lossCum = 0;               
                    	lastCounter = counter;
                	}
                }
                
                // final update
                if(lastCounter != counter) {
                	printLogTime(String.format("%d th iteration, loss: %f (mean time: %d ms) for %d iterations", counter, lossCum / (counter - lastCounter), mean_train_time / (counter - lastCounter),(counter - lastCounter)));
            		List<Weight> new_weight = ParameterEASGDClient.exchangeWeight(host, port, net.getWeights(), movingRate);
                	net.setWeights(new_weight);
                }
            }
        });

        printLogTime("Training complete...");
        server.stopServer();
        net.setGPU();
    }
    public float[] test(List<Record> data) {
    	net.setTestData(net.buildBlobs(data));
        return net.test();
    }
    
    public float[] testWithLMDB() {
    	return net.testWithLMDB();
    }

	public void saveWeight(String pathString) throws FileNotFoundException, IOException {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		Path path = new Path(pathString);
		
		FSDataOutputStream fin = fs.create(path);
		ObjectOutputStream out = new ObjectOutputStream(fin);
		out.writeObject(net.getWeights());
		out.close();
	}

	public void prepareLocal(JavaRDD<Record> data, final String output, int workerSize) {
		JavaRDD<Record> reparted_data = data.repartition(workerSize);
		
		reparted_data.foreach(new VoidFunction<Record>() {
			transient int count;
			transient LMDBWriter dbWriter;
			
			private void readObject(ObjectInputStream in) throws ClassNotFoundException, IOException {
				System.gc();
				in.defaultReadObject();
				File dir = new File(output);
				if(dir.mkdirs())
					printLog("mkdir successfully done...");
				
				dbWriter = new LMDBWriter(output, 1000);
				count = 0;
			}
			
			@Override
			public void call(Record arg0) throws Exception {
				dbWriter.putSample(arg0);
				
				if((++count % 1000) == 0) {
					printLog(String.format("%d images saved...", count));
				}
			}
			
			@Override
			public void finalize() {
				try {
					dbWriter.closeLMDB();
				} catch(Exception e) {
					e.printStackTrace();
				}
				
				printLog(String.format("Total %d images saved...", count));
			}
		});
	}
	
	public void prepareLocalByte(JavaRDD<ByteRecord> data, final String output, int workerSize) {
		JavaRDD<ByteRecord> reparted_data = data.repartition(workerSize);
		
		reparted_data.foreach(new VoidFunction<ByteRecord>() {
			transient int count;
			transient LMDBWriter dbWriter;
			
			private void readObject(ObjectInputStream in) throws ClassNotFoundException, IOException {
				System.gc();
				in.defaultReadObject();
				File dir = new File(output);
				if(dir.mkdirs())
					printLog("mkdir successfully done...");
				
				dbWriter = new LMDBWriter(output, 1000);
				count = 0;
			}
			
			@Override
			public void call(ByteRecord arg0) throws Exception {
				dbWriter.putSample(arg0);
				
				if((++count % 1000) == 0) {
					printLog(String.format("%d images saved...", count));
				}
			}
			
			@Override
			public void finalize() {
				try {
					dbWriter.closeLMDB();
				} catch(Exception e) {
					e.printStackTrace();
				}
				
				printLog(String.format("Total %d images saved...", count));
			}
		});
	}
}
