package org.acl.deepspark.data;

import java.io.Serializable;

public class ByteRecord implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 258491956070013844L;
	
	public byte[] data;
	public float label;
	public int[] dim;
}
