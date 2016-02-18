package org.acl.deepspark.data;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class RecordBatch {
	public ByteBuffer data;
	public ByteBuffer label;
	public int size;
	
	public RecordBatch(int size, int channel, int h, int w) {
		data = ByteBuffer.allocateDirect(Float.SIZE / Byte.SIZE * size * channel * h * w);
		label = ByteBuffer.allocateDirect(Float.SIZE / Byte.SIZE * size);
		this.size = size;
		data.order(ByteOrder.nativeOrder());
		label.order(ByteOrder.nativeOrder());
	}
}
