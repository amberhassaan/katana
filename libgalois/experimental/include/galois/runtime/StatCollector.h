// #ifndef GALOIS_RUNTIME_STATCOLLECTOR_H
#if 0
#define GALOIS_RUNTIME_STATCOLLECTOR_H

#include "galois/gdeque.h"
#include "galois/gIO.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/substrate/PerThreadStorage.h"

#include <string>
#include <set>
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.

namespace galois {
namespace runtime {

class StatCollector {

protected:
  template<typename Ty>
  using StringPair = std::pair<const std::string*, Ty>;

  //////////////////////////////////////
  //Symbol Table
  //////////////////////////////////////
  std::set<std::string> symbols;
  const std::string* getSymbol(const std::string& str) const;
  const std::string* getOrInsertSymbol(const std::string& str);

  //////////////////////////////////////
  //Loop instance counter
  //////////////////////////////////////
  std::vector<StringPair<unsigned> > loopInstances;

  unsigned getInstanceNum(const std::string& str) const;
  void addInstanceNum(const std::string& str);

  //////////////////////////////////////
  //Stat list
  //////////////////////////////////////
  struct RecordTy {
    enum Type {
      INT, DOUBLE, STR
    };

    Type mode;

    union {
      size_t valueInt;
      double valueDouble;
      std::string valueStr;
    };

    RecordTy(size_t value);
    RecordTy(double value);
    RecordTy(const std::string& value);
    RecordTy(const RecordTy& r);
    ~RecordTy();

    void print(std::ostream& out) const;

    size_t intVal(void) const;
    double doubleVal(void) const;
    const std::string& strVal(void) const;
  };



  //stats  HostID,ThreadID,loop,category,instance -> Record
  
  std::string m_outfile;
  std::map<std::tuple<unsigned,unsigned, const std::string*, const std::string*,unsigned>, RecordTy> Stats;
  galois::substrate::SimpleLock StatsLock;

public:

  void printStats(void);

  explicit StatCollector(const std::string& outfile="");

  virtual ~StatCollector(void);

  static boost::uuids::uuid UUID;
  static boost::uuids::uuid getUUID();

  void addToStat(const std::string& loop, const std::string& category, size_t value, unsigned TID, unsigned HostID);
  void addToStat(const std::string& loop, const std::string& category, double value, unsigned TID, unsigned HostID);
  void addToStat(const std::string& loop, const std::string& category, const std::string& value, unsigned TID, unsigned HostID);
  void beginLoopInstance(const std::string& str);

private:
  void printStatsForR(std::ostream& out, bool json);
  void printStats(std::ostream& out);

};


//! Begin a new loop instance
void reportLoopInstance(const char* loopname);
inline void reportLoopInstance(const std::string& loopname) {
  reportLoopInstance(loopname.c_str());
}

//! Reports stats for a given thread
void reportStat(const char* loopname, const char* category, unsigned long value, unsigned TID);
void reportStat(const char* loopname, const char* category, const std::string& value, unsigned TID);
void reportStat(const std::string& loopname, const std::string& category, unsigned long value, unsigned TID);
void reportStat(const std::string& loopname, const std::string& category, const std::string& value, unsigned TID);

void reportStatDist(const std::string& loopname, const std::string& category, const size_t value, unsigned TID, unsigned HostID);
void reportStatDist(const std::string& loopname, const std::string& category, const double value, unsigned TID, unsigned HostID);
void reportStatDist(const std::string& loopname, const std::string& category, const std::string& value, unsigned TID, unsigned HostID);
//! Reports Galois system memory stats for all threads
void reportPageAlloc(const char* category);
//! Reports NUMA memory stats for all NUMA nodes
void reportNumaAlloc(const char* category);

namespace internal {
  void setStatCollector(StatCollector* sc);
}

} // end namespace runtime
} // end namespace galois

#endif
