package org.acl.deepspark.utils;

import java.io.IOException;

import org.acl.deepspark.nn.DistNet;
import org.acl.deepspark.nn.async.ParameterServer;

public class ServerTest {
	public static void main(String args[]) throws IOException {
		DistNet net = new DistNet(args[0], args[1]);
		
		ParameterServer a = new ParameterServer(net, 38470, 38471);
		a.startServer();
		
		
		
		a.stopServer();
	}
}
