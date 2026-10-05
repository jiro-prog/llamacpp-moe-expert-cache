# included after every project() (build.cmd): MinGW names DLLs lib*.dll, but the prebuilt ggml-cuda.dll imports
# "ggml-base.dll" - one ggml-base must serve both, so name ours the MSVC way
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_SHARED_MODULE_PREFIX "")
set(CMAKE_IMPORT_LIBRARY_PREFIX "")
