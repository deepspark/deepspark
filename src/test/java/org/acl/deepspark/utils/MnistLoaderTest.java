package org.acl.deepspark.utils;

public class MnistLoaderTest {
	public static void main(String[] args) {
/*		DoubleMatrix[] mnist_train = MnistLoader.loadData("C:\\Users\\Jaehong\\Downloads\\mnist\\mnist_train.txt");
		System.out.println(String.valueOf(mnist_train.length));

		DoubleMatrix[] mnist_train_label = MnistLoader.loadLabel("C:\\Users\\Jaehong\\Downloads\\mnist\\mnist_train.txt");
		System.out.println(String.valueOf(mnist_train_label.length));
		
		
		DoubleMatrix[] mnist_test = MnistLoader.loadData("C:\\Users\\Jaehong\\Downloads\\mnist\\mnist_test.txt");
		System.out.println(String.valueOf(mnist_test.length));
		
		DoubleMatrix[] mnist_test_label = MnistLoader.loadLabel("C:\\Users\\Jaehong\\Downloads\\mnist\\mnist_test.txt");
		System.out.println(String.valueOf(mnist_test_label.length));
		
		for(int i = 0 ; i < 5;i++)
			System.out.println(mnist_train_label[i]);
		
		for(int i = 0 ; i < 5;i++)
			System.out.println(mnist_test_label[i]);
		
		
		/** MnistLoader Test Complete **/
	}
}
