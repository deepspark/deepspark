package misc;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.DistNet;

import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.SolverParameter;


public class SparkNetWeightConverter {
	public static void convertCaffeModel(String input, String output, String solverSpec,String netSpec) throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(input));
		Map<String, List<float[]>> m = (Map<String, List<float[]>>) in.readObject();
		in.close();
		
		DistNet net = new DistNet(solverSpec,netSpec);
		List<LayerParameter> list = net.getNetConf().getLayerList();
		List<Weight> w = new ArrayList<Weight>();
		for(int i = 0; i < list.size(); i++) {
			LayerParameter p= list.get(i);
			String name = p.getName();
			System.out.println(name);
			
			List<float[]> a = m.get(name);
			if(a != null) {
				for(int k = 0; k < a.size(); k++) {
					Weight d= new Weight();
					d.data = a.get(k);
					d.offset = 0;
					w.add(d);
				}
			}
		}
		
		net.printLearnableWeightInfo();
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
