add_executable(graph-stats graph-stats.cpp)
target_link_libraries(graph-stats PRIVATE katana_graph LLVMSupport)
add_executable(graph-memory-stats graph-memory-stats.cpp)
target_link_libraries(graph-memory-stats PRIVATE katana_graph LLVMSupport)
