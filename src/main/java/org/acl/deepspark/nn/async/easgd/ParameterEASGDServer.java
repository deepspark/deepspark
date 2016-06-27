package org.acl.deepspark.nn.async.easgd;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.DistNet;
import org.acl.deepspark.utils.DeepSparkConf.DeepSparkParam;
import org.acl.deepspark.utils.WeightUtil;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class ParameterEASGDServer {
	private DistNet p;
	private int listenPort;
	private ServerSocket exchangeSocket;
	
	private boolean stopSign = false;
		
	private Thread thread;
	
	private int updateCount = 0;
	
	private int tick;
	private String dirName;
	
	private long startTime;
	
	private int testStep;
	
	private float movingRate;
	private float decayRate;
	private int decayLimit;
	private int decayStep;
	
	private float serverFactor;
	private float factorDecayRate;
	private int factorDecayLimit;
	private int factorDecayStep;
	private PrintStream logStream = null;
	
	
	public void setStartTime(long time) {
		startTime = time;
	}
	
	private void printLogTime(String mesg) {
		if(logStream == null)
			System.out.println(String.format("[%f parameterserver]: %s", (float) (System.currentTimeMillis() - startTime) / 1000 ,mesg));
		else
			logStream.println(String.format("[%f parameterserver]: %s", (float) (System.currentTimeMillis() - startTime) / 1000 ,mesg));
	}
	
	private class EASGDThread implements Runnable {
		private Socket s;
		
		public EASGDThread(Socket s) {
			this.s = s;
		}
		
		@Override
		public void run() {
			try {
				ObjectOutputStream out = new ObjectOutputStream(s.getOutputStream());
				ObjectInputStream in = new ObjectInputStream(s.getInputStream());
				
				out.writeObject(p.getWeights());
				@SuppressWarnings("unchecked")
				List<Weight> client_w = (List<Weight>) in.readObject();
				
				float eff_rate = movingRate * serverFactor;
				printLogTime(String.format("parameter exchanged with coeff %f", eff_rate));
				
				// EASGD update
				List<Weight> server_w = p.getWeights();
				
				WeightUtil.scalarMult(client_w, eff_rate);
				WeightUtil.scalarMult(server_w, 1.0f-eff_rate);
				WeightUtil.addFromTo(client_w, server_w);
				
				p.setWeights(server_w);
				
				s.close();
			} catch (IOException | RuntimeException e) {
				if(stopSign) {
					printLogTime("closed.");
				} else {
					printLogTime("got Error!");
					e.printStackTrace();
				}
			} catch (ClassNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	public ParameterEASGDServer(DistNet net, int listenPort, int numWorker, DeepSparkParam param) {
		p = net;
			
		this.listenPort = listenPort;
		this.startTime = System.currentTimeMillis();
		
		// snapshot
		if(net.getSolverSetting().hasSnapshot()) {
			tick = net.getSolverSetting().getSnapshot() * numWorker / param.getPeriod();
			dirName = net.getSolverSetting().getSnapshotPrefix();
		} else {
			tick = 0;
		}
		
		// intermediate test
		if(p.getSolverSetting().hasTestInterval())
			testStep = p.getSolverSetting().getTestInterval();
		else
			testStep = 0;
						
		// moving rate, alpha
		movingRate = param.getMovingRate();
		decayLimit = param.getDecayLimit();
		decayRate = param.getDecayRate();
		decayStep = param.getDecayStep() * numWorker / param.getPeriod();
		
		// server factor
		serverFactor = param.getServerFactor();
		factorDecayLimit = param.getFactorDecayLimit();
		factorDecayRate = param.getFactorDecayRate();
		factorDecayStep = param.getFactorDecayStep() * numWorker / param.getPeriod();		
	}
		
	public void stopServer() {
		stopSign = true;
		try {
			exchangeSocket.close();
			thread.join();
			logStream.close();
		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
		}
		
	}
	
	public void startServer() throws IOException {
		exchangeSocket = new ServerSocket(listenPort);
		
		// exchange Thread
		thread = new Thread(new Runnable() {
			ExecutorService connectionService;
			
			@Override
			public void run() {
				connectionService = Executors.newFixedThreadPool(8);
				try {
					Configuration conf = new Configuration();
					FileSystem fs;
					fs = FileSystem.get(conf);
					
					// param log init
					if(dirName != null) {
						String logfile = String.format("%s/log.txt", dirName);
						Path path = new Path(logfile);
						FSDataOutputStream fin = fs.create(path);
						logStream = new PrintStream(fin);
					}
					
					while(!stopSign) {
						Socket a = exchangeSocket.accept();
						connectionService.execute(new EASGDThread(a));						
						
						updateCount++;
						
						// intermediate test
						if(testStep != 0 && updateCount % testStep == 0) {
							float[] acc = p.testWithLMDB();
							for(int i = 0;i < acc.length;i++) {
								printLogTime(String.format("test at iteration %d, acc[%d] = %f", updateCount, i, acc[i]));
							}
						}
						
						
						// decay moving rate
						if(decayStep > 0 && decayLimit != 0 && updateCount % decayStep == 0 ) {
							decayLimit--;
							movingRate *= decayRate;
							printLogTime(String.format("moving rate: %f", movingRate));
						}
						
						// decay server factor						
						if(factorDecayStep > 0 && factorDecayLimit != 0 && updateCount % factorDecayStep == 0 ) {
							factorDecayLimit--;
							serverFactor *= factorDecayRate;
							printLogTime(String.format("server factor : %f", serverFactor));
						}
						
						if(tick != 0 && updateCount % tick == 0) {
							String filename = String.format("%s/state_iter_%d.weight", dirName, updateCount);
							Path path = new Path(filename);
							
							FSDataOutputStream fin = fs.create(path);
							ObjectOutputStream out = new ObjectOutputStream(fin);
							out.writeObject(p.getWeights());
							out.close();
							printLogTime(String.format("%s saved..",filename));
						}
					}
				} catch (IOException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
				
				connectionService.shutdown();
				while (!connectionService.isTerminated()) {
		        }
			}
		});
		thread.start();
	}
	
	public int getUpdateCount() {
		return updateCount;
	}
}
