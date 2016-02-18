package org.acl.deepspark.utils;

import java.io.FileReader;
import java.io.IOException;

import org.acl.deepspark.utils.DeepSparkConf.DeepSparkParam;

import com.google.protobuf.TextFormat;

public class DeepSparkParamLoader {
	public static DeepSparkParam readConf(String name) throws IOException {
		FileReader reader = new FileReader(name);
		DeepSparkParam.Builder b = DeepSparkParam.newBuilder();
		TextFormat.merge(reader, b);
		TextFormat.print(b, System.out);
		
		return b.build();
	}
}
