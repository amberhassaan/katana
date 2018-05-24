#include "galois/Galois.h"
#include "galois/AtomicHelpers.h"
#include "galois/Reduction.h"
#include "galois/PriorityQueue.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"
#include "Lonestar/BFS_SSSP.h"

#include <iostream>

namespace cll = llvm::cl;

static const char* name = "Single Source Shortest Path";
static const char* desc =
    "Computes the shortest path from a source node to all nodes in a directed "
    "graph using a modified chaotic iteration algorithm";
static const char* url = "single_source_shortest_path";

static cll::opt<std::string> filename(cll::Positional, 
                                      cll::desc("<input graph>"), 
                                      cll::Required);

static cll::opt<unsigned int> startNode("startNode",
                                        cll::desc("Node to start search from"),
                                        cll::init(0));
static cll::opt<unsigned int> reportNode("reportNode", 
                                         cll::desc("Node to report distance to"),
                                         cll::init(1));
static cll::opt<unsigned int> stepShift("delta",
                               cll::desc("Shift value for the deltastep"),
                               cll::init(13));

enum Algo {
  deltaTile=0,
  deltaStep,
  serDeltaTile,
  serDelta,
  dijkstraTile,
  dijkstra,
  syncTile
};

const char* const ALGO_NAMES[] = {
  "deltaTile",
  "deltaStep",
  "serDeltaTile",
  "serDelta",
  "dijkstraTile",
  "dijkstra",
  "syncTile"
};

static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(deltaTile, "deltaTile", "delta stepping with edge tiling (default)"),
      clEnumValN(deltaStep, "deltaStep", "delta stepping"),
      clEnumValN(serDeltaTile, "serDeltaTile", "serial delta stepping with edge tiling"),
      clEnumValN(serDelta, "serDelta", "serial delta stepping"),
      clEnumValN(dijkstraTile, "dijkstraTile", "dijkstra's algorithm with edge tiling"),
      clEnumValN(dijkstra, "dijkstra", "dijkstra's algorithm"),
      clEnumValN(syncTile, "syncTile", "synchronous edge tiling"),
      clEnumValEnd), cll::init(deltaTile));


const unsigned int DIST_INF = std::numeric_limits<unsigned int>::max()/2 - 1;

struct Node{
  std::atomic<unsigned int> dist_current;
  unsigned int dist_old;
  Node(): dist_current(DIST_INF), dist_old(DIST_INF){}
};


/*using Graph = galois::graphs::LC_CSR_Graph<std::atomic<uint32_t>, uint32_t>
  ::with_no_lockable<true>::type
  ::with_numa_alloc<true>::type;*/
using Graph = galois::graphs::LC_CSR_Graph<Node, uint32_t>
  ::with_no_lockable<true>::type;

typedef Graph::GraphNode GNode;

constexpr static const bool TRACK_WORK = false;
constexpr static const unsigned CHUNK_SIZE = 64u;
constexpr static const ptrdiff_t EDGE_TILE_SIZE = 512;


using SSSP = BFS_SSSP<Graph, uint32_t, true, EDGE_TILE_SIZE>;
using Dist = SSSP::Dist;
using UpdateRequest = SSSP::UpdateRequest;
using UpdateRequestIndexer = SSSP::UpdateRequestIndexer;
using SrcEdgeTile = SSSP::SrcEdgeTile;
using SrcEdgeTileMaker = SSSP::SrcEdgeTileMaker;
using SrcEdgeTilePushWrap = SSSP::SrcEdgeTilePushWrap;
using ReqPushWrap = SSSP::ReqPushWrap;
using OutEdgeRangeFn = SSSP::OutEdgeRangeFn;
using TileRangeFn = SSSP::TileRangeFn;



/*template <typename T, typename P, typename R>
void deltaStepAlgo(Graph& graph, GNode source, const P& pushWrap, const R& edgeRange) {

  galois::GAccumulator<size_t> BadWork;
  galois::GAccumulator<size_t> WLEmptyWork;

  namespace gwl = galois::worklists;

  using dChunk = gwl:: dChunkedFIFO<CHUNK_SIZE>;
  using OBIM = gwl::OrderedByIntegerMetric<UpdateRequestIndexer, dChunk>;

  graph.getData(source) = 0;

  galois::InsertBag<T> initBag;
  pushWrap(initBag, source, 0, "parallel");

  galois::for_each(galois::iterate(initBag), 
      [&] (const T& item, auto& ctx) {

        constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
        const auto& sdata = graph.getData(item.src, flag);
        
        if (sdata < item.dist) {
          if (TRACK_WORK)
            WLEmptyWork += 1;
          return;
        }
        
        for (auto ii: edgeRange(item)) {

          GNode dst = graph.getEdgeDst(ii);
          auto& ddist  = graph.getData(dst, flag);
          Dist ew    = graph.getEdgeData(ii, flag);
          const Dist newDist = sdata + ew;

          while (true) {
            Dist oldDist = ddist;

            if (oldDist <= newDist) { break; }

            if (ddist.compare_exchange_weak(oldDist, newDist, std::memory_order_relaxed)) {

              if (TRACK_WORK) {
                if (oldDist != SSSP::DIST_INFINITY) {
                  BadWork += 1;
                }
              }

              pushWrap(ctx, dst, newDist);
              break;
            }
          }
        }
      },
      galois::wl<OBIM>( UpdateRequestIndexer{stepShift} ), 
      galois::no_conflicts(), 
      galois::loopname("SSSP"));

  if (TRACK_WORK) {
    galois::runtime::reportStat_Single("SSSP", "BadWork", BadWork.reduce());
    galois::runtime::reportStat_Single("SSSP", "WLEmptyWork", WLEmptyWork.reduce()); 
  }
}


template <typename T, typename P, typename R>
void serDeltaAlgo(Graph& graph, const GNode& source, const P& pushWrap, const R& edgeRange) {

  SerialBucketWL<T, UpdateRequestIndexer> wl(UpdateRequestIndexer {stepShift});;
  graph.getData(source) = 0;

  pushWrap(wl, source, 0);

  size_t iter = 0ul;
  while (!wl.empty()) {

    auto& curr = wl.minBucket();

    while (!curr.empty()) {
      ++iter;
      auto item = curr.front();
      curr.pop_front();

      if (graph.getData(item.src) < item.dist) {
        // empty work
        continue;
      }

      for (auto e: edgeRange(item)) {

        GNode dst = graph.getEdgeDst(e);
        auto& ddata = graph.getData(dst);

        const auto newDist = item.dist + graph.getEdgeData(e);

        if (newDist < ddata) {
          ddata = newDist;
          pushWrap(wl, dst, newDist);
        }
      }
    }

    wl.goToNextBucket();
  }

  if (!wl.allEmpty()) { std::abort(); }
  galois::runtime::reportStat_Single("SSSP-Serial-Delta", "Iterations", iter);
}


template <typename T, typename P, typename R>
void dijkstraAlgo(Graph& graph, const GNode& source, const P& pushWrap, const R& edgeRange) {

  using WL = galois::MinHeap<T>;

  graph.getData(source) = 0;

  WL wl;
  pushWrap(wl, source, 0);

  size_t iter = 0;

  while (!wl.empty()) {
    ++iter;

    T item = wl.pop();

    if (graph.getData(item.src) < item.dist) {
      // empty work
      continue;
    }

    for (auto e: edgeRange(item)) {

      GNode dst = graph.getEdgeDst(e);
      auto& ddata = graph.getData(dst);

      const auto newDist = item.dist + graph.getEdgeData(e);

      if (newDist < ddata) {
        ddata = newDist;
        pushWrap(wl, dst, newDist);
      }
    }
  }

  galois::runtime::reportStat_Single("SSSP-Dijkstra", "Iterations", iter);
}*/



void syncAlgo(Graph& graph, GNode source) {

  galois::GAccumulator<unsigned int> changed;
  
  for (auto e: graph.edges(source)) {
    auto dst = graph.getEdgeDst(e);
    auto& ddata = graph.getData(dst);
    const auto newDist = graph.getEdgeData(e);
    const auto oldDist = galois::atomicMin(ddata.dist_current, newDist);
  }

  do
  {
    changed.reset();
    galois::do_all(galois::iterate(graph),
      [&] (const GNode& n) {
        Node& sdata = graph.getData(n);

        if (sdata.dist_old > sdata.dist_current) {
          sdata.dist_old = sdata.dist_current;

          changed += 1;

          for (auto e: graph.edges(n)) {
            auto dst = graph.getEdgeDst(e);
            auto& ddata = graph.getData(dst);
            const auto newDist = sdata.dist_current + graph.getEdgeData(e);

            const auto oldDist = galois::atomicMin(ddata.dist_current, newDist);
          }
        }
      },
      galois::steal(),
      galois::loopname("Sync"));

  }while (changed.reduce());

}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);
  
  Graph graph;
  GNode source, report;

  std::cout << "Reading from file: " << filename << std::endl;
  galois::graphs::readGraph(graph, filename); 
  std::cout << "Read " << graph.size() << " nodes, "
    << graph.sizeEdges() << " edges" << std::endl;

  if (startNode >= graph.size() || reportNode >= graph.size()) {
    std::cerr << "failed to set report: " << reportNode
              << " or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }

  auto it = graph.begin();
  std::advance(it, startNode);
  source = *it;
  it = graph.begin();
  std::advance(it, reportNode);
  report = *it;

  size_t approxNodeData = graph.size() * 64;
  // size_t approxEdgeData = graph.sizeEdges() * sizeof(typename
  // Graph::edge_data_type) * 2;
  galois::preAlloc(numThreads +
                   approxNodeData / galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  std::cout << "INFO: Using delta-step of " << (1 << stepShift) << "\n";
  std::cout << "WARNING: Performance varies considerably due to delta parameter.\n";
  std::cout << "WARNING: Do not expect the default to be good for your graph.\n";
  /*galois::do_all(galois::iterate(graph), 
      [&graph] (GNode n) { 
        graph.getData(n) = SSSP::DIST_INFINITY; 
      });*/

  //graph.getData(source) = 0;
  graph.getData(source).dist_current = 0;
  graph.getData(source).dist_old = DIST_INF;

  std::cout << "Running " <<  ALGO_NAMES[algo] << " algorithm" << std::endl;

  galois::StatTimer Tmain;
  Tmain.start();

  switch(algo) {
    /*case deltaTile:
      deltaStepAlgo<SrcEdgeTile>(graph, source, SrcEdgeTilePushWrap{graph}, TileRangeFn());
      break;
    case deltaStep:
      deltaStepAlgo<UpdateRequest>(graph, source, ReqPushWrap(), OutEdgeRangeFn{graph});
      break;
    case serDeltaTile:
      serDeltaAlgo<SrcEdgeTile>(graph, source, SrcEdgeTilePushWrap{graph}, TileRangeFn());
      break;
    case serDelta:
      serDeltaAlgo<UpdateRequest>(graph, source, ReqPushWrap(), OutEdgeRangeFn{graph});
      break;
    case dijkstraTile:
      dijkstraAlgo<SrcEdgeTile>(graph, source, SrcEdgeTilePushWrap{graph}, TileRangeFn());
      break;
    case dijkstra:
      dijkstraAlgo<UpdateRequest>(graph, source, ReqPushWrap(), OutEdgeRangeFn{graph});
      break;*/

    case syncTile:
      syncAlgo(graph, source);
      break;
    default:
      std::abort();
  }

  Tmain.stop();

  galois::reportPageAlloc("MeminfoPost");
  
  std::cout << "Node " << reportNode << " has distance "
            << graph.getData(report).dist_current << "\n";

  if (!skipVerify) {
    if (SSSP::verify(graph, source)) {
      std::cout << "Verification successful.\n";
    } else {
      GALOIS_DIE("Verification failed");
    }
  }

  return 0;
}
