package org.acl.deepspark.nn.async;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.DistNet;

public class ParameterWorkerServer {
	private DistNet p;
	private int listenPort;
	private int castPort;
	private ServerSocket updateSocket;
	private ServerSocket castSocket;
	
	private boolean stopSign = false;
	private final Object lock = new Object();
	
	private Thread[] threads;
	
	public ParameterWorkerServer(DistNet net, int listenPort, int castPort ) {
		p = net;
		this.listenPort = listenPort;
		this.castPort = castPort;
		threads = new Thread[2];
	}
	
	public void stopServer() {
		stopSign = true;
		try {
			updateSocket.close();
			castSocket.close();
			threads[0].join();
			threads[1].join();
		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	public void startServer() throws IOException {
		updateSocket = new ServerSocket(listenPort);
		castSocket = new ServerSocket(castPort);
		
		// broadcast socket
		threads[0] = new Thread(new Runnable() {
			
			@Override
			public void run() {
				while(!stopSign) {
					try {
						Socket a = castSocket.accept();
						synchronized (lock) {
							ObjectOutputStream os = new ObjectOutputStream(a.getOutputStream());
							os.writeObject(p.getWeights());
						}
						System.out.println("ParameterServer: sent!");
						a.close();
					} catch (IOException | RuntimeException e) {
						if(stopSign) {
							System.out.println("ParameterServer[broadcast]: closed.");
						} else {
							System.out.println("ParameterServer[broadcast]: got Error!");
							e.printStackTrace();
						}
					}
				}
			}
		});
		threads[0].start();
		
		// update socket
		threads[1] = new Thread(new Runnable() {
			
			@Override
			public void run() {
				// TODO Auto-generated method stub
				while(!stopSign) {
					try {
						Socket a = updateSocket.accept();
						synchronized (lock) {
							ObjectInputStream is = new ObjectInputStream(a.getInputStream());
							List<Weight> w = (List<Weight>) is.readObject();
							p.setGradients(w);
							p.manualUpdate();
							System.out.println("ParameterServer: Updated!");
						}
						a.close();
					} catch (IOException | ClassNotFoundException | RuntimeException e) {
						if(stopSign) {
							System.out.println("ParameterServer[update]: closed.");
						} else {
							System.out.println("ParameterServer[update]: got Error!");
							e.printStackTrace();
						}
					}
				}
			}
		});
		threads[1].start();	
	}
}
