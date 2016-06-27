package org.acl.deepspark.nn.async;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.util.List;

import org.acl.deepspark.data.Weight;

public class ParameterClient {
	public static void sendDelta(String host, int port, List<Weight> d) throws IOException, ClassNotFoundException {
		Socket s = new Socket(host, port);
		s.setSoTimeout(15000);
		ObjectOutputStream os = new ObjectOutputStream(s.getOutputStream());
		os.writeObject(d);
		s.close();
		System.out.println("Gradients sent");
	}
	
	public static List<Weight> getWeights(String host, int port) throws IOException, ClassNotFoundException {
		Socket s = new Socket(host, port);
		s.setSoTimeout(15000);
		ObjectInputStream os = new ObjectInputStream(s.getInputStream());
		List<Weight> w = (List<Weight>) os.readObject();
		s.close();
		System.out.println("Weight received");
		return w;
	}
}
