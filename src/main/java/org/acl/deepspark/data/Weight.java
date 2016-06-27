package org.acl.deepspark.data;

import java.io.Serializable;

public class Weight implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public float[] data;
	public int offset;
	public int layerIndex;
}
