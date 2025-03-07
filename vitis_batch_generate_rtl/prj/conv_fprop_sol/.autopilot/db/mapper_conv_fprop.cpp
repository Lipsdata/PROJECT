#include "hls_signal_handler.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <vector>
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_directio.h"
#include "hls_stream.h"
using namespace std;

namespace hls::sim
{
  template<size_t n>
  struct Byte {
    unsigned char a[n];

    Byte()
    {
      for (size_t i = 0; i < n; ++i) {
        a[i] = 0;
      }
    }

    template<typename T>
    Byte<n>& operator= (const T &val)
    {
      std::memcpy(a, &val, n);
      return *this;
    }
  };

  struct SimException : public std::exception {
    const std::string msg;
    const size_t line;
    SimException(const std::string &msg, const size_t line)
      : msg(msg), line(line)
    {
    }
  };

  void errExit(const size_t line, const std::string &msg)
  {
    std::string s;
    s += "ERROR";
//  s += '(';
//  s += __FILE__;
//  s += ":";
//  s += std::to_string(line);
//  s += ')';
    s += ": ";
    s += msg;
    s += "\n";
    fputs(s.c_str(), stderr);
    exit(1);
  }
}


namespace hls::sim
{
  struct Buffer {
    char *first;
    Buffer(char *addr) : first(addr)
    {
    }
  };

  struct DBuffer : public Buffer {
    static const size_t total = 1<<10;
    size_t ufree;

    DBuffer(size_t usize) : Buffer(nullptr), ufree(total)
    {
      first = new char[usize*ufree];
    }

    ~DBuffer()
    {
      delete[] first;
    }
  };

  struct CStream {
    char *front;
    char *back;
    size_t num;
    size_t usize;
    std::list<Buffer*> bufs;
    bool dynamic;

    CStream() : front(nullptr), back(nullptr),
                num(0), usize(0), dynamic(true)
    {
    }

    ~CStream()
    {
      for (Buffer *p : bufs) {
        delete p;
      }
    }

    template<typename T>
    T* data()
    {
      return (T*)front;
    }

    template<typename T>
    void transfer(hls::stream<T> *param)
    {
      while (!empty()) {
        param->write(*(T*)nextRead());
      }
    }

    bool empty();
    char* nextRead();
    char* nextWrite();
  };

  bool CStream::empty()
  {
    return num == 0;
  }

  char* CStream::nextRead()
  {
    assert(num > 0);
    char *res = front;
    front += usize;
    if (dynamic) {
      if (++static_cast<DBuffer*>(bufs.front())->ufree == DBuffer::total) {
        if (bufs.size() > 1) {
          bufs.pop_front();
          front = bufs.front()->first;
        } else {
          front = back = bufs.front()->first;
        }
      }
    }
    --num;
    return res;
  }

  char* CStream::nextWrite()
  {
    if (dynamic) {
      if (static_cast<DBuffer*>(bufs.back())->ufree == 0) {
        bufs.push_back(new DBuffer(usize));
        back = bufs.back()->first;
      }
      --static_cast<DBuffer*>(bufs.back())->ufree;
    }
    char *res = back;
    back += usize;
    ++num;
    return res;
  }

  std::list<CStream> streams;
  std::map<char*, CStream*> prebuilt;

  CStream* createStream(size_t usize)
  {
    streams.emplace_front();
    CStream &s = streams.front();
    {
      s.dynamic = true;
      s.bufs.push_back(new DBuffer(usize));
      s.front = s.bufs.back()->first;
      s.back = s.front;
      s.num = 0;
      s.usize = usize;
    }
    return &s;
  }

  template<typename T>
  CStream* createStream(hls::stream<T> *param)
  {
    CStream *s = createStream(sizeof(T));
    {
      s->dynamic = true;
      while (!param->empty()) {
        T data = param->read();
        memcpy(s->nextWrite(), (char*)&data, sizeof(T));
      }
      prebuilt[s->front] = s;
    }
    return s;
  }

  template<typename T>
  CStream* createStream(T *param, size_t usize)
  {
    streams.emplace_front();
    CStream &s = streams.front();
    {
      s.dynamic = false;
      s.bufs.push_back(new Buffer((char*)param));
      s.front = s.back = s.bufs.back()->first;
      s.usize = usize;
      s.num = ~0UL;
    }
    prebuilt[s.front] = &s;
    return &s;
  }

  CStream* findStream(char *buf)
  {
    return prebuilt.at(buf);
  }
}
class AESL_RUNTIME_BC {
  public:
    AESL_RUNTIME_BC(const char* name) {
      file_token.open( name);
      if (!file_token.good()) {
        cout << "Failed to open tv file " << name << endl;
        exit (1);
      }
      file_token >> mName;//[[[runtime]]]
    }
    ~AESL_RUNTIME_BC() {
      file_token.close();
    }
    int read_size () {
      int size = 0;
      file_token >> mName;//[[transaction]]
      file_token >> mName;//transaction number
      file_token >> mName;//pop_size
      size = atoi(mName.c_str());
      file_token >> mName;//[[/transaction]]
      return size;
    }
  public:
    fstream file_token;
    string mName;
};
using hls::sim::Byte;
struct __cosim_s400__ { char data[400]; };
struct __cosim_s512__ { char data[512]; };
extern "C" void conv_fprop(volatile void *, volatile void *, volatile void *, Byte<8>*, Byte<8>*, Byte<8>*, Byte<8>*, volatile void *, volatile void *, volatile void *, Byte<512>*, Byte<8>*, volatile void *, volatile void *, volatile void *, Byte<8>*, Byte<8>*, Byte<8>*, Byte<8>*, volatile void *, volatile void *, volatile void *, Byte<512>*, Byte<8>*, Byte<1>*);
extern "C" void apatb_conv_fprop_hw(volatile void * __xlx_apatb_param_prev_layer_map_w, volatile void * __xlx_apatb_param_prev_layer_map_h, volatile void * __xlx_apatb_param_prev_layer_map_count, volatile void * __xlx_apatb_param_prev_layer_map_data, volatile void * __xlx_apatb_param_prev_layer_map_error, volatile void * __xlx_apatb_param_prev_layer_map_b, volatile void * __xlx_apatb_param_prev_layer_map_db, volatile void * __xlx_apatb_param_prev_layer_kernel_w, volatile void * __xlx_apatb_param_prev_layer_kernel_h, volatile void * __xlx_apatb_param_prev_layer_kernel_count, volatile void * __xlx_apatb_param_prev_layer_kernel, volatile void * __xlx_apatb_param_prev_layer_map_common, volatile void * __xlx_apatb_param_layer_map_w, volatile void * __xlx_apatb_param_layer_map_h, volatile void * __xlx_apatb_param_layer_map_count, volatile void * __xlx_apatb_param_layer_map_data, volatile void * __xlx_apatb_param_layer_map_error, volatile void * __xlx_apatb_param_layer_map_b, volatile void * __xlx_apatb_param_layer_map_db, volatile void * __xlx_apatb_param_layer_kernel_w, volatile void * __xlx_apatb_param_layer_kernel_h, volatile void * __xlx_apatb_param_layer_kernel_count, volatile void * __xlx_apatb_param_layer_kernel, volatile void * __xlx_apatb_param_layer_map_common, volatile void * __xlx_apatb_param_pconnection) {
using hls::sim::createStream;
  // Collect __xlx_prev_layer_map_data__tmp_vec
std::vector<Byte<8>> __xlx_prev_layer_map_data__tmp_vec;
for (size_t i = 0; i < 122880; ++i){
__xlx_prev_layer_map_data__tmp_vec.push_back(((Byte<8>*)__xlx_apatb_param_prev_layer_map_data)[i]);
}
  int __xlx_size_param_prev_layer_map_data = 122880;
  int __xlx_offset_param_prev_layer_map_data = 0;
  int __xlx_offset_byte_param_prev_layer_map_data = 0*8;
  // Collect __xlx_prev_layer_map_error__tmp_vec
std::vector<Byte<8>> __xlx_prev_layer_map_error__tmp_vec;
for (size_t i = 0; i < 122880; ++i){
__xlx_prev_layer_map_error__tmp_vec.push_back(((Byte<8>*)__xlx_apatb_param_prev_layer_map_error)[i]);
}
  int __xlx_size_param_prev_layer_map_error = 122880;
  int __xlx_offset_param_prev_layer_map_error = 0;
  int __xlx_offset_byte_param_prev_layer_map_error = 0*8;
  // Collect __xlx_prev_layer_map_b__tmp_vec
std::vector<Byte<8>> __xlx_prev_layer_map_b__tmp_vec;
for (size_t i = 0; i < 120; ++i){
__xlx_prev_layer_map_b__tmp_vec.push_back(((Byte<8>*)__xlx_apatb_param_prev_layer_map_b)[i]);
}
  int __xlx_size_param_prev_layer_map_b = 120;
  int __xlx_offset_param_prev_layer_map_b = 0;
  int __xlx_offset_byte_param_prev_layer_map_b = 0*8;
  // Collect __xlx_prev_layer_map_db__tmp_vec
std::vector<Byte<8>> __xlx_prev_layer_map_db__tmp_vec;
for (size_t i = 0; i < 120; ++i){
__xlx_prev_layer_map_db__tmp_vec.push_back(((Byte<8>*)__xlx_apatb_param_prev_layer_map_db)[i]);
}
  int __xlx_size_param_prev_layer_map_db = 120;
  int __xlx_offset_param_prev_layer_map_db = 0;
  int __xlx_offset_byte_param_prev_layer_map_db = 0*8;
  // Collect __xlx_prev_layer_kernel__tmp_vec
std::vector<Byte<512>> __xlx_prev_layer_kernel__tmp_vec;
for (size_t i = 0; i < 1920; ++i){
__xlx_prev_layer_kernel__tmp_vec.push_back(((Byte<512>*)__xlx_apatb_param_prev_layer_kernel)[i]);
}
  int __xlx_size_param_prev_layer_kernel = 1920;
  int __xlx_offset_param_prev_layer_kernel = 0;
  int __xlx_offset_byte_param_prev_layer_kernel = 0*512;
  // Collect __xlx_prev_layer_map_common__tmp_vec
std::vector<Byte<8>> __xlx_prev_layer_map_common__tmp_vec;
for (size_t i = 0; i < 1024; ++i){
__xlx_prev_layer_map_common__tmp_vec.push_back(((Byte<8>*)__xlx_apatb_param_prev_layer_map_common)[i]);
}
  int __xlx_size_param_prev_layer_map_common = 1024;
  int __xlx_offset_param_prev_layer_map_common = 0;
  int __xlx_offset_byte_param_prev_layer_map_common = 0*8;
  // Collect __xlx_layer_map_data__tmp_vec
std::vector<Byte<8>> __xlx_layer_map_data__tmp_vec;
for (size_t i = 0; i < 122880; ++i){
__xlx_layer_map_data__tmp_vec.push_back(((Byte<8>*)__xlx_apatb_param_layer_map_data)[i]);
}
  int __xlx_size_param_layer_map_data = 122880;
  int __xlx_offset_param_layer_map_data = 0;
  int __xlx_offset_byte_param_layer_map_data = 0*8;
  // Collect __xlx_layer_map_error__tmp_vec
std::vector<Byte<8>> __xlx_layer_map_error__tmp_vec;
for (size_t i = 0; i < 122880; ++i){
__xlx_layer_map_error__tmp_vec.push_back(((Byte<8>*)__xlx_apatb_param_layer_map_error)[i]);
}
  int __xlx_size_param_layer_map_error = 122880;
  int __xlx_offset_param_layer_map_error = 0;
  int __xlx_offset_byte_param_layer_map_error = 0*8;
  // Collect __xlx_layer_map_b__tmp_vec
std::vector<Byte<8>> __xlx_layer_map_b__tmp_vec;
for (size_t i = 0; i < 120; ++i){
__xlx_layer_map_b__tmp_vec.push_back(((Byte<8>*)__xlx_apatb_param_layer_map_b)[i]);
}
  int __xlx_size_param_layer_map_b = 120;
  int __xlx_offset_param_layer_map_b = 0;
  int __xlx_offset_byte_param_layer_map_b = 0*8;
  // Collect __xlx_layer_map_db__tmp_vec
std::vector<Byte<8>> __xlx_layer_map_db__tmp_vec;
for (size_t i = 0; i < 120; ++i){
__xlx_layer_map_db__tmp_vec.push_back(((Byte<8>*)__xlx_apatb_param_layer_map_db)[i]);
}
  int __xlx_size_param_layer_map_db = 120;
  int __xlx_offset_param_layer_map_db = 0;
  int __xlx_offset_byte_param_layer_map_db = 0*8;
  // Collect __xlx_layer_kernel__tmp_vec
std::vector<Byte<512>> __xlx_layer_kernel__tmp_vec;
for (size_t i = 0; i < 1920; ++i){
__xlx_layer_kernel__tmp_vec.push_back(((Byte<512>*)__xlx_apatb_param_layer_kernel)[i]);
}
  int __xlx_size_param_layer_kernel = 1920;
  int __xlx_offset_param_layer_kernel = 0;
  int __xlx_offset_byte_param_layer_kernel = 0*512;
  // Collect __xlx_layer_map_common__tmp_vec
std::vector<Byte<8>> __xlx_layer_map_common__tmp_vec;
for (size_t i = 0; i < 1024; ++i){
__xlx_layer_map_common__tmp_vec.push_back(((Byte<8>*)__xlx_apatb_param_layer_map_common)[i]);
}
  int __xlx_size_param_layer_map_common = 1024;
  int __xlx_offset_param_layer_map_common = 0;
  int __xlx_offset_byte_param_layer_map_common = 0*8;
  // Collect __xlx_pconnection__tmp_vec
std::vector<Byte<1>> __xlx_pconnection__tmp_vec;
for (size_t i = 0; i < 96; ++i){
__xlx_pconnection__tmp_vec.push_back(((Byte<1>*)__xlx_apatb_param_pconnection)[i]);
}
  int __xlx_size_param_pconnection = 96;
  int __xlx_offset_param_pconnection = 0;
  int __xlx_offset_byte_param_pconnection = 0*1;
  // DUT call
  conv_fprop(__xlx_apatb_param_prev_layer_map_w, __xlx_apatb_param_prev_layer_map_h, __xlx_apatb_param_prev_layer_map_count, __xlx_prev_layer_map_data__tmp_vec.data(), __xlx_prev_layer_map_error__tmp_vec.data(), __xlx_prev_layer_map_b__tmp_vec.data(), __xlx_prev_layer_map_db__tmp_vec.data(), __xlx_apatb_param_prev_layer_kernel_w, __xlx_apatb_param_prev_layer_kernel_h, __xlx_apatb_param_prev_layer_kernel_count, __xlx_prev_layer_kernel__tmp_vec.data(), __xlx_prev_layer_map_common__tmp_vec.data(), __xlx_apatb_param_layer_map_w, __xlx_apatb_param_layer_map_h, __xlx_apatb_param_layer_map_count, __xlx_layer_map_data__tmp_vec.data(), __xlx_layer_map_error__tmp_vec.data(), __xlx_layer_map_b__tmp_vec.data(), __xlx_layer_map_db__tmp_vec.data(), __xlx_apatb_param_layer_kernel_w, __xlx_apatb_param_layer_kernel_h, __xlx_apatb_param_layer_kernel_count, __xlx_layer_kernel__tmp_vec.data(), __xlx_layer_map_common__tmp_vec.data(), __xlx_pconnection__tmp_vec.data());
// print __xlx_apatb_param_prev_layer_map_data
for (size_t i = 0; i < __xlx_size_param_prev_layer_map_data; ++i) {
((Byte<8>*)__xlx_apatb_param_prev_layer_map_data)[i] = __xlx_prev_layer_map_data__tmp_vec[__xlx_offset_param_prev_layer_map_data+i];
}
// print __xlx_apatb_param_prev_layer_map_error
for (size_t i = 0; i < __xlx_size_param_prev_layer_map_error; ++i) {
((Byte<8>*)__xlx_apatb_param_prev_layer_map_error)[i] = __xlx_prev_layer_map_error__tmp_vec[__xlx_offset_param_prev_layer_map_error+i];
}
// print __xlx_apatb_param_prev_layer_map_b
for (size_t i = 0; i < __xlx_size_param_prev_layer_map_b; ++i) {
((Byte<8>*)__xlx_apatb_param_prev_layer_map_b)[i] = __xlx_prev_layer_map_b__tmp_vec[__xlx_offset_param_prev_layer_map_b+i];
}
// print __xlx_apatb_param_prev_layer_map_db
for (size_t i = 0; i < __xlx_size_param_prev_layer_map_db; ++i) {
((Byte<8>*)__xlx_apatb_param_prev_layer_map_db)[i] = __xlx_prev_layer_map_db__tmp_vec[__xlx_offset_param_prev_layer_map_db+i];
}
// print __xlx_apatb_param_prev_layer_kernel
for (size_t i = 0; i < __xlx_size_param_prev_layer_kernel; ++i) {
((Byte<512>*)__xlx_apatb_param_prev_layer_kernel)[i] = __xlx_prev_layer_kernel__tmp_vec[__xlx_offset_param_prev_layer_kernel+i];
}
// print __xlx_apatb_param_prev_layer_map_common
for (size_t i = 0; i < __xlx_size_param_prev_layer_map_common; ++i) {
((Byte<8>*)__xlx_apatb_param_prev_layer_map_common)[i] = __xlx_prev_layer_map_common__tmp_vec[__xlx_offset_param_prev_layer_map_common+i];
}
// print __xlx_apatb_param_layer_map_data
for (size_t i = 0; i < __xlx_size_param_layer_map_data; ++i) {
((Byte<8>*)__xlx_apatb_param_layer_map_data)[i] = __xlx_layer_map_data__tmp_vec[__xlx_offset_param_layer_map_data+i];
}
// print __xlx_apatb_param_layer_map_error
for (size_t i = 0; i < __xlx_size_param_layer_map_error; ++i) {
((Byte<8>*)__xlx_apatb_param_layer_map_error)[i] = __xlx_layer_map_error__tmp_vec[__xlx_offset_param_layer_map_error+i];
}
// print __xlx_apatb_param_layer_map_b
for (size_t i = 0; i < __xlx_size_param_layer_map_b; ++i) {
((Byte<8>*)__xlx_apatb_param_layer_map_b)[i] = __xlx_layer_map_b__tmp_vec[__xlx_offset_param_layer_map_b+i];
}
// print __xlx_apatb_param_layer_map_db
for (size_t i = 0; i < __xlx_size_param_layer_map_db; ++i) {
((Byte<8>*)__xlx_apatb_param_layer_map_db)[i] = __xlx_layer_map_db__tmp_vec[__xlx_offset_param_layer_map_db+i];
}
// print __xlx_apatb_param_layer_kernel
for (size_t i = 0; i < __xlx_size_param_layer_kernel; ++i) {
((Byte<512>*)__xlx_apatb_param_layer_kernel)[i] = __xlx_layer_kernel__tmp_vec[__xlx_offset_param_layer_kernel+i];
}
// print __xlx_apatb_param_layer_map_common
for (size_t i = 0; i < __xlx_size_param_layer_map_common; ++i) {
((Byte<8>*)__xlx_apatb_param_layer_map_common)[i] = __xlx_layer_map_common__tmp_vec[__xlx_offset_param_layer_map_common+i];
}
// print __xlx_apatb_param_pconnection
for (size_t i = 0; i < __xlx_size_param_pconnection; ++i) {
((Byte<1>*)__xlx_apatb_param_pconnection)[i] = __xlx_pconnection__tmp_vec[__xlx_offset_param_pconnection+i];
}
}
