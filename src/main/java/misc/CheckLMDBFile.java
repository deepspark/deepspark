package misc;

import org.acl.deepspark.data.LMDBWriter;

public class CheckLMDBFile {
	public static void main(String[] args) {
		String path = args[0];
		
		LMDBWriter w = new LMDBWriter(path);
		w.printAll();
		
	}
}
