package misc;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.List;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.DistNet;

import caffe.Caffe.NetParameter;
import caffe.Caffe.SolverParameter;

public class WeightConverter {
	public static void convertCaffeModel(String input, String output, String solverSpec,String netSpec) throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(input));
		List<Weight> w = (List<Weight>) in.readObject();
		in.close();
		
		DistNet net = new DistNet(solverSpec,netSpec);
		net.setWeights(w);
		net.snapshot(output);
	}
	
	public static void main(String[] args) throws FileNotFoundException, ClassNotFoundException, IOException {
		String inName = args[0];
		String outName = args[1];
		String solSpec = args[2];
		String netSpec = args[3];
		
		convertCaffeModel(inName, outName, solSpec, netSpec);
	}
}
