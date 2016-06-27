package misc;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

import caffe.Caffe;
import caffe.Caffe.BlobProto;

public class PrintDeltaGradient {

	public static void main(String[] args) throws FileNotFoundException, IOException {
		// TODO Auto-generated method stub
		String listname = args[0];
		BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(listname)));
		
		String filename;
		while((filename = in.readLine()) != null) {
			Caffe.SolverState solver = Caffe.SolverState.parseFrom(new FileInputStream(filename));
			
			List<BlobProto> data = solver.getHistoryList();
			int count = 0;
			float sum = 0;
			
			for(BlobProto d : data) {
				List<Float> g = d.getDataList();
				for(Float a : g) {
					sum+= a.floatValue() * a.floatValue();
				}
				count += g.size();
			}
			
			System.out.println(String.format("%d\t%f", count, sum));
		}
		
		in.close();
	}

}
