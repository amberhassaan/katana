/*
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include <arrow/type_traits.h>

#include "katana/ArrowVisitor.h"
#include "katana/Galois.h"
#include "katana/JSON.h"
#include "katana/Logging.h"
#include "katana/OfflineGraph.h"
#include "llvm/Support/CommandLine.h"
#include "tsuba/Errors.h"
#include "tsuba/FaultTest.h"
#include "tsuba/FileView.h"

namespace cll = llvm::cl;
static cll::opt<std::string> inputfilename(
    cll::Positional, cll::desc("graph-file"), cll::Required);

static cll::opt<std::string> outputfilename(
    cll::Positional, cll::desc("out-file"), cll::Required);

enum class GraphElement: int {
  kNode = 0,
  kEdge
};

template <typename T>
struct Wrapper {
  T val_;
  explicit Wrapper(const T& v) noexcept : val_(v) {}
  operator const T& () const noexcept { return val_; }
  operator T& () noexcept { return val_; }
};

struct MetaDataSize: public Wrapper<size_t> {
  using Wrapper<size_t>::Wrapper; // imports constructors
};

struct DataSize: public Wrapper<size_t> {
  using Wrapper<size_t>::Wrapper; // imports constructors
};

struct PropertyMemUsage {
  DataSize data_size_;
  MetaDataSize meta_data_size_;
  
  PropertyMemUsage(const DataSize& d, const MetaDataSize& m) noexcept:
    data_size_(d),
    meta_data_size_(m) {}

  size_t totalBytes() const noexcept {
    return size_t(data_size_) + size_t(meta_data_size_);
  }

  void add(const PropertyMemUsage& that) noexcept {
    static_cast<size_t&>(this->data_size_) += static_cast<const size_t&>(that.data_size_);
    static_cast<size_t&>(this->meta_data_size_) += static_cast<const size_t&>(that.meta_data_size_);
  }
};


struct WithoutGrouping: public PropertyMemUsage {
  using PropertyMemUsage::PropertyMemUsage;
};

struct WithGrouping: public PropertyMemUsage {
  using PropertyMemUsage::PropertyMemUsage;
};

struct OptimizedGrouping: public PropertyMemUsage {
  using PropertyMemUsage::PropertyMemUsage;
};

struct PropertyName: public Wrapper<std::string> {
  using Wrapper<std::string>::Wrapper;
};

struct TypeName: public Wrapper<std::string> {
  using Wrapper<std::string>::Wrapper;
};

struct PropertyStat {
  PropertyName prop_name_;
  TypeName type_name_;
  GraphElement elem_kind_;
  WithoutGrouping without_grouping_;
  WithGrouping with_grouping_;
  OptimizedGrouping optimized_grouping_;

  PropertyStat(
      const PropertyName& name, 
      const TypeName& type,
      const GraphElement& node_or_edge,
      const WithoutGrouping& no_grp, 
      const WithGrouping& w_grp, 
      const OptimizedGrouping& opt_grp) noexcept:
    prop_name_(name),
    type_name_(type),
    elem_kind_(node_or_edge),
    without_grouping_(no_grp),
    with_grouping_(w_grp),
    optimized_grouping_(opt_grp) {}


  std::string key() const noexcept {
    return static_cast<std::string>(prop_name_) + 
      (elem_kind_ == GraphElement::kNode ? "-Node-" : "-Edge-") + 
      static_cast<std::string>(type_name_);
  }

  void add(const PropertyStat& that) noexcept {
    KATANA_LOG_DEBUG_ASSERT(this->key() == that.key());
    this->without_grouping_.add(that.without_grouping_);
    this->with_grouping_.add(that.with_grouping_);
    this->optimized_grouping_.add(that.optimized_grouping_);
  }
};

using PropertyStatMap = std::unordered_map<std::string, PropertyStat>;

using MemoryUsageMap = std::unordered_map<std::string, int64_t>;
using TypeInformationMap = std::unordered_map<std::string, std::string>;
using FullReportMap = std::unordered_map<
    std::string, std::variant<MemoryUsageMap, TypeInformationMap>>;

struct ArrayVisitor : public katana::ArrowVisitor {
  using ResultType = katana::Result<std::tuple<WithoutGrouping, WithGrouping, OptimizedGrouping>>;
  using AcceptTypes = std::tuple<katana::AcceptAllFlatTypes>;

  template <typename ArrowType, typename ArrayType>
  arrow::enable_if_null<ArrowType, ResultType> Call(const ArrayType& prop_arr) {
    return KATANA_ERROR(
        katana::ErrorCode::ArrowError, "can't analyze type {}",
        prop_arr.type()->ToString());
  }

  template <typename ArrowType, typename ArrayType>
  std::enable_if_t<
      arrow::is_number_type<ArrowType>::value ||
          arrow::is_boolean_type<ArrowType>::value ||
          arrow::is_temporal_type<ArrowType>::value,
      ResultType>
  Call(const ArrayType& prop_arr) {

    using ElemType = typename ArrayType::TypeClass;
    size_t num_valid = 0;
    for (auto j = 0; j < prop_arr.length(); j++) {
      if (!prop_arr.IsNull(j)) {
        ++num_valid;
      }
    }

    size_t real_used_space = num_valid * sizeof(ElemType);
    size_t space_allocated = sizeof(ElemType) * prop_arr.length();

    fmt::print("Space (bytes) using dense layout : {}, using sparse layout = {}\n", 
        space_allocated, real_used_space);
    size_t null_bitmap_size = prop_arr.null_bitmap()->size();

    fmt::print("null_bitmap_data = {}\n", prop_arr.null_bitmap()->size());

    WithoutGrouping without_grouping{ DataSize{space_allocated}, MetaDataSize{null_bitmap_size}};
    WithGrouping with_grouping{ DataSize{real_used_space}, MetaDataSize{null_bitmap_size}};
    OptimizedGrouping opt_grouping{ DataSize{real_used_space}, MetaDataSize{(num_valid + 8 - 1) / 8u}};

    return std::make_tuple(without_grouping, with_grouping, opt_grouping);
  }

  template <typename ArrowType, typename ArrayType>
  arrow::enable_if_string_like<ArrowType, ResultType> Call(
      const ArrayType& prop_arr) {
    using OffsetType = typename ArrayType::offset_type;
    size_t tot_len = prop_arr.total_values_length();
    size_t metadata_size = sizeof(OffsetType) * prop_arr.length();
    size_t null_bitmap_size = prop_arr.null_bitmap()->size();

    metadata_size += null_bitmap_size;

    size_t num_valid = 0;
    for (auto j = 0; j < prop_arr.length(); j++) {
      if (!prop_arr.IsNull(j)) {
        ++num_valid;
      }
    }

    // sum of two terms. 
    // - First is the size of the offset array. 
    // - Second is the size of the null_bitmap
    size_t opt_meta_data = num_valid * sizeof(OffsetType) + num_valid / 8u;

    WithoutGrouping without_grouping{ DataSize{tot_len}, MetaDataSize{metadata_size}};
    WithGrouping with_grouping{ DataSize{tot_len}, MetaDataSize{metadata_size}};
    OptimizedGrouping opt_grouping{ DataSize{tot_len}, MetaDataSize{opt_meta_data}};

    return std::make_tuple(without_grouping, with_grouping, opt_grouping);

  }

  ResultType AcceptFailed(const arrow::Array& prop_arr) {
    return KATANA_ERROR(
        katana::ErrorCode::ArrowError, "no matching type {}",
        prop_arr.type()->ToString());
  }
};


PropertyStat
AnalyzeProperty(
    const std::shared_ptr<arrow::Array> prop_arr, const std::string prop_name,
    const std::string& type_name, const GraphElement& node_or_edge) {

  ArrayVisitor v;
  auto res = katana::VisitArrow(v, *prop_arr.get());
  KATANA_LOG_VASSERT(res, "unexpected errror {}", res.error());

  auto tup =  res.value();
  return PropertyStat {
      PropertyName{prop_name},
      TypeName{type_name},
      node_or_edge,
      std::get<0>(tup),
      std::get<1>(tup),
      std::get<2>(tup)};

}

void 
RecordStat(PropertyStatMap& mem_usage_stats, const PropertyStat& stat) noexcept {
  if (auto it = mem_usage_stats.find(stat.key()); it != mem_usage_stats.end()) {
    it->second.add(stat);
  } else {
    mem_usage_stats.insert(it, std::make_pair(stat.key(), stat));
  }
}

template <GraphElement ELEM_KIND>
void
GatherPropertyMemStats(
    const katana::PropertyGraph* graph,
    const std::shared_ptr<arrow::Schema> schema,
    PropertyStatMap& mem_usage_stats) noexcept {

  for (int32_t i = 0; i < schema->num_fields(); ++i) {
    const std::string& prop_name = schema->field(i)->name();
    const std::shared_ptr<arrow::DataType>& dtype = schema->field(i)->type();

    fmt::print("Property Name: {}, type = {} \n", prop_name, dtype->ToString());

    if (dtype->id() == arrow::Type::UINT8) {
      continue;
    }

    std::shared_ptr<arrow::Array> prop_arr;
    if (ELEM_KIND == GraphElement::kNode) {
      prop_arr = graph->GetNodeProperty(prop_name).value()->chunk(0);
    } else {
      prop_arr = graph->GetEdgeProperty(prop_name).value()->chunk(0);
    }

    RecordStat(mem_usage_stats, AnalyzeProperty(
        prop_arr, prop_name, dtype->name(), ELEM_KIND));

  }
}

void AddMetaDataStat(PropertyStatMap& mem_usage_stats,
    const PropertyName& prop_name, const TypeName& type_name,
    const GraphElement& node_or_edge,
    const MetaDataSize& meta_data) noexcept {

  DataSize d{0};
  RecordStat(mem_usage_stats, 
    PropertyStat{
      prop_name,
      type_name,
      node_or_edge,
      WithoutGrouping{ d, meta_data},
      WithGrouping{ d, meta_data},
      OptimizedGrouping{ d, meta_data}
    });
}

void 
GatherTopologyStats(const katana::PropertyGraph* graph, PropertyStatMap& mem_usage_stats) noexcept {

  fmt::print("Graph has {} Nodes and {} Edges\n", graph->num_nodes(), graph->num_edges());
  
  // Adj Indices
  AddMetaDataStat(
      mem_usage_stats,
      PropertyName{"adj_indices"},
      TypeName{"uint64_t"},
      GraphElement::kNode,
      MetaDataSize{graph->num_nodes() * sizeof(katana::GraphTopology::Edge)});

  // Edge Destinations
  AddMetaDataStat(
      mem_usage_stats,
      PropertyName{"edge_dests"},
      TypeName{"uint32_t"},
      GraphElement::kEdge,
      MetaDataSize{graph->num_edges() * sizeof(katana::GraphTopology::Edge)});


  // Node type info
  AddMetaDataStat(
      mem_usage_stats,
      PropertyName{"node_type_ids"},
      TypeName{"uint8_t"},
      GraphElement::kNode,
      MetaDataSize{graph->num_nodes() * sizeof(katana::EntityTypeID)});

  // Edge type info
  AddMetaDataStat(
      mem_usage_stats,
      PropertyName{"edge_type_ids"},
      TypeName{"uint8_t"},
      GraphElement::kEdge,
      MetaDataSize{graph->num_edges() * sizeof(katana::EntityTypeID)});

}

PropertyStatMap
DoMemoryAnalysis(const katana::PropertyGraph* graph, PropertyStatMap& mem_usage_stats) {

  fmt::print("Gathering Topology Stats: \n");
  GatherTopologyStats(graph, mem_usage_stats);

  fmt::print("Gathering Node Stats: \n");
  GatherPropertyMemStats<GraphElement::kNode>(
      graph, graph->full_node_schema(), mem_usage_stats);

  fmt::print("Gathering Edge Stats: \n");
  GatherPropertyMemStats<GraphElement::kEdge>(
      graph, graph->full_edge_schema(), mem_usage_stats);


  return mem_usage_stats;
}

void
WriteCSV(const PropertyStatMap& mem_usage_stats, const std::string& out_file_name) noexcept {

  std::ofstream fh(out_file_name);
  KATANA_LOG_ASSERT(fh.good());

  // header
  fh << "Property Name, Node_or_Edge, Data Type, Without Grouping(bytes), With Grouping(bytes), Optimized Grouping(bytes)" << std::endl;

  size_t tot_no_grp = 0;
  size_t tot_with_grp = 0;
  size_t tot_opt_grp = 0;

  for (const auto& [k, p]: mem_usage_stats) {
    fh 
      << static_cast<const std::string&>(p.prop_name_)
      << ", "
      << ((p.elem_kind_ == GraphElement::kNode) ? "Node" : "Edge")
      << ", "
      << static_cast<const std::string&>(p.type_name_)
      << ", "
      << p.without_grouping_.totalBytes()
      << ", "
      << p.with_grouping_.totalBytes()
      << ", "
      << p.optimized_grouping_.totalBytes()
      << std::endl;

    tot_no_grp += p.without_grouping_.totalBytes();
    tot_with_grp += p.with_grouping_.totalBytes();
    tot_opt_grp += p.optimized_grouping_.totalBytes();
  }

  fh 
    << "Total"
    << ", "
    << "Graph"
    << ", "
    << "All-types"
    << ", "
    << tot_no_grp
    << ", "
    << tot_with_grp
    << ", "
    << tot_opt_grp
    << ", "
    << std::endl;

  fh.close();

}

/*
katana::Result<void>
SaveToJson(
    const katana::Result<std::string>& json_to_dump,
    const std::string& out_path, const std::string& name_extension) {
  std::ofstream myfile;
  std::string path_to_save = out_path + name_extension;

  if (!json_to_dump) {
    return json_to_dump.error();
  }

  std::string serialized(json_to_dump.value());
  serialized = serialized + "\n";

  auto ff = std::make_unique<tsuba::FileFrame>();
  if (auto res = ff->Init(serialized.size()); !res) {
    return res.error();
  }

  if (auto res = ff->Write(serialized.data(), serialized.size()); !res.ok()) {
    return KATANA_ERROR(
        tsuba::ArrowToTsuba(res.code()), "arrow error: {}", res);
  }

  myfile.open(path_to_save);
  myfile << serialized;
  myfile.close();

  return katana::ResultSuccess();
}

void
doMemoryAnalysis(const katana::PropertyGraph* graph) {

  fmt::print("Num Nodes = {}, Num Edges = {}\n", graph->num_nodes(), graph->num_edges());


  FullReportMap mem_map = {};
  MemoryUsageMap basic_raw_stats = {};
  auto node_schema = graph->full_node_schema();
  auto edge_schema = graph->full_edge_schema();
  int64_t total_num_node_props = node_schema->num_fields();
  int64_t total_num_edge_props = edge_schema->num_fields();

  fmt::print("Number of Node Properties = {}\n", total_num_node_props);
  fmt::print("Number of Edge Properties = {}\n", total_num_edge_props);

  basic_raw_stats.insert(std::pair("Node-Schema-Size", total_num_node_props));
  basic_raw_stats.insert(std::pair("Edge-Schema-Size", total_num_edge_props));
  basic_raw_stats.insert(
      std::pair("Number-Node-Atomic-Types", graph->GetNumNodeAtomicTypes()));
  basic_raw_stats.insert(
      std::pair("Number-Edge-Atomic-Types", graph->GetNumEdgeAtomicTypes()));
  basic_raw_stats.insert(
      std::pair("Number-Node-Entity-Types", graph->GetNumNodeEntityTypes()));
  basic_raw_stats.insert(
      std::pair("Number-Edge-Entity-Types", graph->GetNumNodeEntityTypes()));
  basic_raw_stats.insert(std::pair("Number-Nodes", graph->num_nodes()));
  basic_raw_stats.insert(std::pair("Number-Edges", graph->num_edges()));

  auto atomic_node_types = graph->ListAtomicNodeTypes();
  auto atomic_edge_types = graph->ListAtomicEdgeTypes();

  TypeInformationMap all_node_prop_stats;
  TypeInformationMap all_edge_prop_stats;
  MemoryUsageMap all_node_width_stats;
  MemoryUsageMap all_edge_width_stats;
  MemoryUsageMap all_node_alloc;
  MemoryUsageMap all_edge_alloc;
  MemoryUsageMap all_node_usage;
  MemoryUsageMap all_edge_usage;

  fmt::print("Printing Node Stats: \n");

  GatherPropertyMemStats<GraphElement::kNode>(
      node_schema, graph, &all_node_alloc, &all_node_usage,
      &all_node_width_stats, &all_node_prop_stats);

  mem_map.insert(std::pair("Node-Types", all_node_prop_stats));

  fmt::print("Printing Edge Stats: \n");

  GatherPropertyMemStats<GraphElement::kEdge>(
      edge_schema, graph, &all_edge_alloc, &all_edge_usage,
      &all_edge_width_stats, &all_edge_prop_stats);

  mem_map.insert(std::pair("Edge-Types", all_edge_prop_stats));

  mem_map.insert(std::pair("General-Stats", basic_raw_stats));

  auto basic_raw_json_res = SaveToJson(
      katana::JsonDump(basic_raw_stats), outputfilename,
      "basic_raw_stats.json");
  KATANA_LOG_VASSERT(
      basic_raw_json_res, "unexpected errror {}", basic_raw_json_res.error());

  auto all_node_prop_json_res = SaveToJson(
      katana::JsonDump(all_node_prop_stats), outputfilename,
      "node_prop_stats.json");
  KATANA_LOG_VASSERT(
      all_node_prop_json_res, "unexpected errror {}",
      all_node_prop_json_res.error());

  auto all_node_width_json_res = SaveToJson(
      katana::JsonDump(all_node_width_stats), outputfilename,
      "node_width_stats.json");
  KATANA_LOG_VASSERT(
      all_node_width_json_res, "unexpected errror {}",
      all_node_width_json_res.error());

  auto all_edge_prop_json_res = SaveToJson(
      katana::JsonDump(all_edge_prop_stats), outputfilename,
      "edge_prop_stats.json");
  KATANA_LOG_VASSERT(
      all_edge_prop_json_res, "unexpected errror {}",
      all_edge_prop_json_res.error());

  auto all_edge_width_json_res = SaveToJson(
      katana::JsonDump(all_edge_width_stats), outputfilename,
      "edge_width_stats.json");
  KATANA_LOG_VASSERT(
      all_edge_width_json_res, "unexpected errror {}",
      all_edge_width_json_res.error());

  auto all_node_alloc_json_res = SaveToJson(
      katana::JsonDump(all_node_alloc), outputfilename,
      "default_node_alloc.json");
  KATANA_LOG_VASSERT(
      all_node_alloc_json_res, "unexpected errror {}",
      all_node_alloc_json_res.error());

  auto all_edge_alloc_json_res = SaveToJson(
      katana::JsonDump(all_edge_alloc), outputfilename,
      "default_edge_alloc.json");
  KATANA_LOG_VASSERT(
      all_edge_alloc_json_res, "unexpected errror {}",
      all_edge_alloc_json_res.error());

  auto all_node_usage_json_res = SaveToJson(
      katana::JsonDump(all_node_usage), outputfilename,
      "grouping_node_usage.json");
  KATANA_LOG_VASSERT(
      all_node_usage_json_res, "unexpected errror {}",
      all_node_usage_json_res.error());

  auto all_edge_usage_json_res = SaveToJson(
      katana::JsonDump(all_edge_usage), outputfilename,
      "grouping_edge_usage.json");
  KATANA_LOG_VASSERT(
      all_edge_usage_json_res, "unexpected errror {}",
      all_edge_usage_json_res.error());
      
}
*/


uint64_t GetNumPartitions(const std::string& rdg_dir) {

  auto rdg_view_res = tsuba::ListViewsOfVersion(rdg_dir);
  KATANA_LOG_ASSERT(rdg_view_res);
  std::vector<tsuba::RDGView> views = rdg_view_res.value().second;
  KATANA_LOG_ASSERT(views.size() > 0);

  return views[0].num_partitions;
  
}

int
main(int argc, char** argv) {
  katana::SharedMemSys sys;
  llvm::cl::ParseCommandLineOptions(argc, argv);


  auto num_part = GetNumPartitions(inputfilename);
  fmt::print("Number of partitions = {}\n", num_part);

  PropertyStatMap mem_usage_stats;

  for (uint32_t part_id = 0; part_id < num_part; ++part_id) {
    auto pg_res = katana::PropertyGraph::Make(inputfilename, tsuba::RDGLoadOptions{.partition_id_to_load = part_id});
    KATANA_LOG_ASSERT(pg_res);
    std::unique_ptr<katana::PropertyGraph> pg(std::move(pg_res.value()));

    DoMemoryAnalysis(pg.get(), mem_usage_stats);
  }

  fmt::print("Writing results to CSV\n");
  WriteCSV(mem_usage_stats, outputfilename);

  return 0;
}
