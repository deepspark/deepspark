package org.acl.deepspark.utils;

import java.util.Arrays;
import java.util.List;

import org.acl.deepspark.data.Weight;

public class WeightUtil {
	public static void addFromTo(List<Weight> from, List<Weight> to) {
		if(from.size() != to.size()) {
			throw new RuntimeException("Size mismatch!");
		}
		
		for(int i = 0; i < from.size();i++) {
			Weight f = from.get(i);
			Weight t = to.get(i);
			
			for(int j = f.offset; j < f.data.length; j++) {
				t.data[j] += f.data[j-f.offset];
			}
		}	
	}
	
	public static void sub(List<Weight> sub, List<Weight> inPlace) {
		if(sub.size() != inPlace.size()) {
			throw new RuntimeException("Size mismatch!");
		}
		
		for(int i = 0; i < sub.size();i++) {
			Weight f = sub.get(i);
			Weight t = inPlace.get(i);
			
			for(int j = f.offset; j < f.data.length; j++) {
				t.data[j] -= f.data[j-f.offset];
			}
		}	
	}
	
	public static void scalarMult(List<Weight> to, float ratio) {
		for(int i = 0; i < to.size();i++) {
			Weight t = to.get(i);
			
			for(int j = 0; j < t.data.length; j++) {
				t.data[j] *= ratio;
			}
		}	
	}	
	
	public static void clear(List<Weight> l) {
		for(int i =0; i < l.size(); i++) {
			Arrays.fill(l.get(i).data, 0);
		}
	}
}
