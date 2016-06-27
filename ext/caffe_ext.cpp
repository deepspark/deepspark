#include "caffe_ext.hpp"
#include <string>
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/caffe.hpp"
#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/sgd_solvers.hpp"
#include "caffe/util/io.hpp"

//#include "glog/logging.h"
//#include "caffe/util/io.hpp"
//#include "caffe/util/db.hpp"

//#define GET_DATUM_PTR(datum) reinterpret_cast<Datum*>(datum)
#define GET_SOLVER_PTR(solver) reinterpret_cast<DistSGDSolver<float>* >(solver)

using namespace caffe;
using namespace std;

template <typename Dtype>
class DistSGDSolver : public caffe::SGDSolver<Dtype> {
	public:
		explicit DistSGDSolver(const caffe::SolverParameter& param)
			: caffe::SGDSolver<Dtype>(param) { }
		explicit DistSGDSolver(const std::string& param_file)
			: caffe::SGDSolver<Dtype>(param_file) { }

		virtual Dtype singleTrain();
		virtual void update();
		virtual void* test();
		virtual void getGradient(Dtype* dst, int offset, int num, int pid);
		virtual void getWeight(Dtype* dst, int offset, int num, int pid);
		virtual void setGradient(Dtype* src, int offset, int num, int pid);
		virtual void setWeight(Dtype* src, int offset, int num, int pid);
		virtual const int getNumWeight(int pid)const;
		virtual const int getNumLearnableParams() const;

		virtual void snapshotNet(const char* name) const;
		virtual void restoreNet(const char* name);

		DISABLE_COPY_AND_ASSIGN(DistSGDSolver);
};


// JNA function start!
/*
void writeDatum2Lmdb(void* arr, char* lmdbPath) {
	Datum* datumArr = GET_DATUM_PTR(arr);
	std::vector<Datum> data(std::begin(datumArr), std::end(datumArr));

	// Create a new DB
	scoped_ptr<db::DB> db(db::GetDB("lmdb"));
	db->Open(lmdbPath, db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());


	// Storing to db
	int count = 0;
	int data_size = 0;
	bool data_size_initialized = false;
	for(int line_id = 0; line_id < data.size(); ++line_id) {
		Datum datum = data[line_id];
		if (!data_size_initialized) {
			data_size = datum.channels() * datum.height() * datum.width();
			data_size_initialized = true;
		} else {
			const std::string& str = datum.data();
			CHECK_EQ(str.size(), data_size) << "Incorrect data field size " << data.size();
		}
		// sequential
		string key = caffe::format_int(line_id, 8);

		// Put in db
		string out;
		CHECK(datum.SerializeToString(&out));
		txn->Put(key, out);

		if (++count % 1000 == 0) {
			// Commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(INFO) << "Processed " << count << " files.";
		}
	}
	// write the last batch
	if (count % 1000 != 0) {
		txn->Commit();
		LOG(INFO) << "Processed " << count << "files.";
	}
}
*/

void* allocateSolver(const char* solverSpec, int solverSpecLength, const char* netSpec, int netSpecLength) {
	std::string solverParamSer(solverSpec, solverSpecLength);
	std::string netParamSer(netSpec, netSpecLength);

	SolverParameter solverParam;
	solverParam.ParseFromString(solverParamSer);
	NetParameter* netParam = new NetParameter();
	netParam->ParseFromString(netParamSer);

	solverParam.set_allocated_net_param(netParam);

	return reinterpret_cast<void*>(new DistSGDSolver<float>(solverParam));
}

void releaseSolver(void* solver) {
	delete GET_SOLVER_PTR(solver);
}

void setTrainBuffer(void* solver, const char* data_layer_name, void* data, void* label, int num) {
	DistSGDSolver<float>* sol = GET_SOLVER_PTR(solver);
	
	shared_ptr<Net<float> > net = sol->net();
	shared_ptr<MemoryDataLayer<float> > train_data_layer = boost::static_pointer_cast<MemoryDataLayer<float> >(net->layer_by_name(data_layer_name));
	
	train_data_layer->Reset(static_cast<float*>(data), static_cast<float*>(label) ,num);
}

void setTrainData(void* solver, const char* data_layer_name, float* data, float* label, int num) {
	DistSGDSolver<float>* sol = GET_SOLVER_PTR(solver);
	
	shared_ptr<Net<float> > net = sol->net();
	shared_ptr<MemoryDataLayer<float> > train_data_layer = boost::static_pointer_cast<MemoryDataLayer<float> >(net->layer_by_name(data_layer_name));
	
	int channel =  train_data_layer->channels();
	int h = train_data_layer->height();
	int w = train_data_layer->width();

	float* trainer_data = new float[num*channel*h*w];
	float* trainer_label = new float[num];

	memcpy(trainer_data, data, sizeof(float)* num*channel*h*w);
	memcpy(trainer_label, label, sizeof(float)*num);

	train_data_layer->Reset(trainer_data, trainer_label ,num);
}

void setTestBuffer(void* solver, int net_index, const char* data_layer_name, void* data, void* label, int num) {
	DistSGDSolver<float>* sol = GET_SOLVER_PTR(solver);
	
	shared_ptr<Net<float> > test_net = sol->test_nets()[net_index];
	shared_ptr<MemoryDataLayer<float> > test_data_layer = boost::static_pointer_cast<MemoryDataLayer<float> >(test_net->layer_by_name(data_layer_name));

	test_data_layer->Reset(static_cast<float*>(data), static_cast<float*>(label) ,num);
}

void setTestData(void* solver, int net_index, const char* data_layer_name, float* data, float* label, int num) {
	DistSGDSolver<float>* sol = GET_SOLVER_PTR(solver);
	
	shared_ptr<Net<float> > test_net = sol->test_nets()[net_index];
	shared_ptr<MemoryDataLayer<float> > test_data_layer = boost::static_pointer_cast<MemoryDataLayer<float> >(test_net->layer_by_name(data_layer_name));
	
	int channel =  test_data_layer->channels();
	int h = test_data_layer->height();
	int w = test_data_layer->width();

	float* tester_data = new float[num*channel*h*w];
	float* tester_label = new float[num];

	memcpy(tester_data, data, sizeof(float)* num*channel*h*w);
	memcpy(tester_label, label, sizeof(float)*num);

	test_data_layer->Reset(tester_data, tester_label ,num);
}

int getNumWeight(void* solver, int pid) {
	return GET_SOLVER_PTR(solver)->getNumWeight(pid);
}

int getNumLearnableLayer(void* solver) {
	return GET_SOLVER_PTR(solver)->getNumLearnableParams();
}

void getGradient(void* solver, float* buf, int offset, int num, int pid) {
	GET_SOLVER_PTR(solver)->getGradient(buf, offset,num, pid);
}

void setGradient(void* solver, float* params, int offset, int num, int pid) {
	GET_SOLVER_PTR(solver)->setGradient(params,offset,num, pid);
}

void getWeight(void* solver, float* buf, int offset, int num, int pid) {
	GET_SOLVER_PTR(solver)->getWeight(buf,offset,num,  pid);
}

void setWeight(void* solver, float* params, int offset, int num, int pid) {
	GET_SOLVER_PTR(solver)->setWeight(params,offset,num, pid);
}

float train(void* solver) {
	return GET_SOLVER_PTR(solver)->singleTrain();
}

void update(void* solver) {
	GET_SOLVER_PTR(solver)->update();
}

void* test(void* solver) {
	return GET_SOLVER_PTR(solver)->test();
}

void set_mode(int data) {
	Caffe::set_mode(static_cast<caffe::Caffe::Brew>(data));
}

int testCuda() {
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	return nDevices;
}

void destroy(void* p) {
	free(p);
}

void snapshotNet(void* p, const char* name) {
	GET_SOLVER_PTR(p)->snapshotNet(name);
}

void restoreNet(void* p, const char* name) {
	GET_SOLVER_PTR(p)->restoreNet(name);
}

// class implementation
template <typename Dtype>
Dtype DistSGDSolver<Dtype>::singleTrain() {
	Dtype loss;
	loss = this->net()->ForwardBackward();

	Dtype rate = this->GetLearningRate();
	this->ClipGradients();
	for (int param_id = 0; param_id < this->net_->learnable_params().size();
			++param_id) {
		this->Normalize(param_id);
		this->Regularize(param_id);
		this->ComputeUpdateValue(param_id, rate);
	}

	this->update();
	this->iter_++;
	return loss;
}

template <typename Dtype>
void DistSGDSolver<Dtype>::update() {
		this->net()->Update();
}

template <typename Dtype>
void DistSGDSolver<Dtype>::getGradient(Dtype* dst, int offset, int num, int pid) {
	const vector<Blob<Dtype>* >& param= this->net_->learnable_params();
	const Dtype* ptr_param= param[pid]->cpu_diff();
	for(int i =0; i < num; i++)
		dst[i] = ptr_param[i+offset];
}

template <typename Dtype>
void DistSGDSolver<Dtype>::setGradient(Dtype* params, int offset, int num, int pid) {
	const vector<Blob<Dtype>* >& param= this->net_->learnable_params();
	Dtype* ptr_param= param[pid]->mutable_cpu_diff();
	for(int i =0; i < num; i++)
		ptr_param[i+offset] = params[i];
}

template <typename Dtype>
void DistSGDSolver<Dtype>::getWeight(Dtype* dst, int offset, int num, int pid) {
	const vector<Blob<Dtype>* >& param= this->net_->learnable_params();
	const Dtype* ptr_param= param[pid]->cpu_data();
	for(int i =0; i < num; i++)
		dst[i] = ptr_param[i+offset];
}

template <typename Dtype>
void DistSGDSolver<Dtype>::setWeight(Dtype* params, int offset, int num, int pid) {
	const vector<Blob<Dtype>* >& param= this->net_->learnable_params();
	Dtype* ptr_param= param[pid]->mutable_cpu_data();
	for(int i =0; i < num; i++)
		ptr_param[i+offset] = params[i];
}

template <typename Dtype>
const int DistSGDSolver<Dtype>::getNumLearnableParams() const {
	return this->net_->learnable_params().size();
}

template <typename Dtype>
const int DistSGDSolver<Dtype>::getNumWeight(int pid) const {
	const vector<Blob<Dtype>* >& param= this->net_->learnable_params();
	return param[pid]->count();
}

template <typename Dtype>
void* DistSGDSolver<Dtype>::test() {
  this->test_nets_[0].get()->ShareTrainedLayersWith(this->net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = this->test_nets_[0];
  Dtype loss = 0;

  for (int i = 0; i < this->param_.test_iter(0); ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(bottom_vec, &iter_loss);
    if (this->param_.test_compute_loss()) {
      loss += iter_loss;
    }

    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }

  if (this->param_.test_compute_loss()) {
    loss /= this->param_.test_iter(0);
  }

  void* ret = malloc(sizeof(Dtype) * test_score.size() + sizeof(int));
  Dtype* res = static_cast<Dtype*>(ret + sizeof(int));

  *((int*)ret) = test_score.size();

  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    Dtype mean_score = test_score[i] / this->param_.test_iter(0);
    if (loss_weight) {
		mean_score *= loss_weight;
    }
	res[i] = mean_score;
  }
  return reinterpret_cast<void*>(ret);
}

template <typename Dtype>
void DistSGDSolver<Dtype>::snapshotNet(const char* name) const {
	string filename(name);
	
	NetParameter net_param;
	this->net_->ToProto(&net_param, false);
	WriteProtoToBinaryFile(net_param, filename);
}

template <typename Dtype>
void DistSGDSolver<Dtype>::restoreNet(const char* name) {
	string filename(name);
	this->net_->CopyTrainedLayersFromBinaryProto(filename);
}
