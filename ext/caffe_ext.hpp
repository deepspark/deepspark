#ifndef __ACL_CAFFE_EXT_HPP
#define __ACL_CAFFE_EXT_HPP

// exposed by JNA
extern "C" {	
//	void writeDatum2Lmdb(void* arr, char* lmdbPath);


	void* allocateSolver(const char* solverSpec, int, const char* netSpec, int);
	void releaseSolver(void* solver);

	void setTrainData(void* solver, const char* data_layer_name, float* data, float* label, int num);
	void setTrainBuffer(void* solver, const char* data_layer_name, void* data, void* label, int num);

	void setTestData(void* solver, int net_index, const char* data_layer_name, float* data, float* label, int num);
	void setTestBuffer(void* solver, int net_index, const char* data_layer_name, void* data, void* label, int num);

	int getNumWeight(void* solver, int pid);
	int getNumLearnableLayer(void* solver);

	void getGradient(void* solver, float* dest, int offset, int num, int pid);
	void setGradient(void* solver, float* params, int offset, int num, int pid);
	void getWeight(void* solver, float* dest,int offset, int num, int pid);
	void setWeight(void* solver, float* params, int offset, int num, int pid);

	int testCuda();

	float train(void* solver);
	void update(void* solver);
	void* test(void* solver);
	void destroy(void* ptr);

	void set_mode(int data);

	void snapshotNet(void* solver, const char* name);
	void restoreNet(void* solver, const char* name);
}
#endif
