package org.acl.deepspark.nn.async;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import org.acl.deepspark.nn.DistNet;

public class ParameterMasterServer {
	private DistNet p;
	private int port;
	private CyclicBarrier barrier;
	private ServerSocket serverSocket;
	
	private List<Thread> threads;
	private ParameterRange[] paramRanges;
	private String[] hostnames;
	
	
	private class RegisterThread extends Thread {
		private Socket connect;
		private int idx;
		public RegisterThread(Socket s, int idx) {
			connect = s;
			this.idx= idx;
		}
		
		@Override
		public void run() {
			try {
				BufferedReader in = new BufferedReader(new InputStreamReader(connect.getInputStream())); 
				OutputStream out = connect.getOutputStream();

				//setup hostname
				String hostName = in.readLine();
				hostnames[idx] = hostName;
				
				// push map index
				out.write(idx);
				
				// wait other workers
				barrier.await();
				
				// push map
				ObjectOutputStream objOut = new ObjectOutputStream(out);
				objOut.writeObject(paramRanges);
				
				// push hostnames
				objOut.writeObject(hostnames);
				
			} catch (InterruptedException | BrokenBarrierException | IOException e) {
				System.out.println("barrier failed.....");
				e.printStackTrace();
			}
		}
		
	}
	
	public ParameterMasterServer(DistNet net, int port, int barrierSize) {
		p = net;
		this.port = port;
		this.barrier = new CyclicBarrier(barrierSize);
		
	}
	
	public void acceptRegister() throws IOException {
		serverSocket = new ServerSocket(port);
		
		//prepare paramMap
		paramRanges = new ParameterRange[barrier.getParties()];
		hostnames = new String[barrier.getParties()];
		p.getWeights();
		
		
		//
		for(int i =0; i < barrier.getParties() ; i++) {
			Socket s = serverSocket.accept();
			RegisterThread t = new RegisterThread(s, i);
			t.start();
			threads.add(t);
		}
		
		for(int i = 0; i < threads.size(); i++) {
			try {
				threads.get(i).join();
			} catch (InterruptedException e) {
				System.err.println(String.format("Thread %d is interrupted..", i));
				e.printStackTrace();
			}
		}
	}
}
