CXX=g++

CUDA_INC=/usr/local/cuda/include
CAFFE_DIR=../caffe

OBJS=caffe_ext.o
TARGET=libcaffeext.so
INCLUDES=-I $(CAFFE_DIR)/include -I $(CUDA_INC) -I $(CAFFE_DIR)/build/src
LDFLAGS=-fPIC -shared -L$(CAFFE_DIR)/build/lib
#LIBS=-Wl,-Bstatic libcaffe.a -lglog -lprotobuf -Wl,-Bdynamic -lm
LIBS=-lcaffe
CXXFLAGS=$(INCLUDES) -fPIC -DPIC 

.SUFFIXES : .cpp .o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(CAFFE_OBJS) $(LIBS)

clean:
	rm -f $(OBJS) $(TARGET)
