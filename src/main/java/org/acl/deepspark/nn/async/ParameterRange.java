package org.acl.deepspark.nn.async;

import java.io.Serializable;

public class ParameterRange implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1315785049722464754L;
	private int size;
	private int[] offset;
	private int[] length;
	
	public ParameterRange(int size) {
		this.size = size;
		this.offset = new int[size];
		this.length = new int[size];
	}

	public int getSize() {
		return size;
	}

	public int[] getOffset() {
		return offset;
	}

	public int[] getLength() {
		return length;
	}
	
	public void set(int layerId, int offset, int length) {
		this.offset[layerId] = offset;
		this.length[layerId] = length;
	}
}
