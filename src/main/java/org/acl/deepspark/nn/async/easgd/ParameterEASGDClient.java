package org.acl.deepspark.nn.async.easgd;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.util.List;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.utils.WeightUtil;

public class ParameterEASGDClient {
	public static List<Weight> exchangeWeight(String host, int port, List<Weight> client_w, float coeff) throws IOException, ClassNotFoundException {
		long start = System.currentTimeMillis();
		Socket s = new Socket(host, port);
		s.setSoTimeout(1500000);
		
		ObjectOutputStream out = new ObjectOutputStream(s.getOutputStream());
		ObjectInputStream in = new ObjectInputStream(s.getInputStream());
		
		@SuppressWarnings("unchecked")
		List<Weight> server_w = (List<Weight>) in.readObject();
		System.out.println("client: weight recieved");
		
		out.writeObject(client_w);
		System.out.println("client: weight sent");
		s.close();
		
		
		WeightUtil.scalarMult(server_w, coeff);
		WeightUtil.scalarMult(client_w, 1.0f-coeff);
		WeightUtil.addFromTo(server_w, client_w);
		
		long end = System.currentTimeMillis();
		System.out.println(String.format("client: weight exchanged (%d ms)", end-start));
		return client_w;
	}
}
