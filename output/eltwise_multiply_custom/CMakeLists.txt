add_library(ttnn_eltwise_multiply_custom ${LIB_TYPE})
add_library(TT::NN::Ops::EltwiseMultiplyCustom ALIAS ttnn_eltwise_multiply_custom)

target_precompile_headers(ttnn_eltwise_multiply_custom REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_eltwise_multiply_custom)

target_sources(
    ttnn_eltwise_multiply_custom
    PRIVATE
        eltwise_multiply_custom.cpp
        eltwise_multiply_custom_pybind.cpp
        device/eltwise_multiply_custom_op.cpp
        device/eltwise_multiply_custom_program_factory.cpp
)

target_include_directories(ttnn_eltwise_multiply_custom PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_eltwise_multiply_custom
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_eltwise_multiply_custom LIBRARY COMPONENT tar)