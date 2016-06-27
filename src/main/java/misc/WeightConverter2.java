package misc;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.List;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.DistNet;

public class WeightConverter2 {
	public static void convertCaffeModel(String inName, String output, String solverSpec1,String netSpec1) throws FileNotFoundException, IOException, ClassNotFoundException {
		DistNet net = new DistNet(solverSpec1,netSpec1);
		net.restore(inName);
		List<Weight> w = net.getWeights();
		
		ObjectOutputStream objout = new ObjectOutputStream(new FileOutputStream(output));
		objout.writeObject(w);
		objout.close();
	}
	
	public static void main(String[] args) throws FileNotFoundException, ClassNotFoundException, IOException {
		String inName = args[0];
		String outName = args[1];
		String solSpec1 = args[2];
		String netSpec1 = args[3];
		
		convertCaffeModel(inName, outName, solSpec1, netSpec1);
	}
}
