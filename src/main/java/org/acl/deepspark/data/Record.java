package org.acl.deepspark.data;

import java.io.Serializable;

public class Record implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 258491956070013844L;
	
	public float[] data;
	public float label;
	public int[] dim;
}
