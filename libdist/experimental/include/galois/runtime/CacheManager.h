#ifndef GALOIS_RUNTIME_CACHEMANAGER_H
#define GALOIS_RUNTIME_CACHEMANAGER_H

#include "galois/substrate/SimpleLock.h"
#include "galois/runtime/FatPointer.h"
//#include "galois/runtime/Serialize.h"

#include <unordered_map>
#include <deque>
#include <mutex>

namespace galois {
namespace runtime {

namespace internal {
struct dser_t {};
struct tser_t {};
}

class DeSerializeBuffer;
template<typename T>
T gDeserializeObj(DeSerializeBuffer&);

namespace internal {

class remoteObj {
  std::atomic<unsigned> refs;
public:
  void incRef() { ++refs; }
  void decRef() { --refs; }
  virtual ~remoteObj();
  virtual void* getObj() = 0;
};

template<typename T>
class remoteObjImpl : public remoteObj {
  T obj;
public:
  //  remoteObjImpl(DeSerializeBuffer& buf) :obj{std::move(gDeserializeObj<T>(buf))} {}
  template<typename U = void>
  remoteObjImpl(DeSerializeBuffer& buf, internal::dser_t* p) :obj{buf} {}
  template<typename U = void>
  remoteObjImpl(DeSerializeBuffer& buf, internal::tser_t* p) {
    gDeserialize(buf, obj);
  }

  remoteObjImpl(T&& buf) :obj(buf) {}
  virtual ~remoteObjImpl() {}
  virtual void* getObj() { return &obj; }
};

}

class ResolveCache {
  std::unordered_map<fatPointer, void*> addrs;
  std::deque<internal::remoteObj*> objs;
 
public:
  ~ResolveCache() { reset(); }
  void* resolve(fatPointer);
  void reset();
};


class CacheManager {
  std::unordered_map<fatPointer, internal::remoteObj*> remoteObjects;
  LL::SimpleLock Lock;
  std::deque<internal::remoteObj*> garbage;

  //! resolve a value to a reference count incremented metadata
  //! used by ResolveCache
  internal::remoteObj* resolveIncRef(fatPointer);

  friend class ResolveCache;
  friend class RemoteDirectory;

public:

  template<typename T>
  void create(fatPointer ptr, DeSerializeBuffer& buf) {
    assert(ptr.getHost() != NetworkInterface::ID);
    std::lock_guard<LL::SimpleLock> lgr(Lock);
    internal::remoteObj*& obj = remoteObjects[ptr];
    if (obj) { // creating can replace old objects
      garbage.push_back(obj);
    }
    //FIXME: need to do RO lock
    typename std::conditional<has_serialize<T>::value, internal::dser_t, internal::tser_t>::type* P = 0;
    obj = new internal::remoteObjImpl<T>(buf, P);
  }

  template<typename T>
  void create(fatPointer ptr, T&& buf) {
    assert(ptr.getHost() != NetworkInterface::ID);
    std::lock_guard<LL::SimpleLock> lgr(Lock);
    internal::remoteObj*& obj = remoteObjects[ptr];
    if (obj) { // creating can replace old objects
      garbage.push_back(obj);
    }
    //FIXME: need to do RO lock
    obj = new internal::remoteObjImpl<T>(std::forward<T>(buf));
  }

  //! Does not write back. That must be handled by the directory
  void evict(fatPointer);

  //! resolve a value, mainly (always?) for serial code
  void* resolve(fatPointer);

  bool isCurrent(fatPointer, void*);

  //! Get the size of remoteObject map
  size_t CM_size();
};

ResolveCache* getThreadResolve();
void setThreadResolve(ResolveCache*);

CacheManager& getCacheManager();

}
}

#endif
