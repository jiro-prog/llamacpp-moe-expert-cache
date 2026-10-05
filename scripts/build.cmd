@echo off
rem Shared-library build of llama.cpp b11361 (commit a4cb4c61f, %QWEN_ROOT%\src\llama.cpp-a4cb4c61f, patched by setup.ps1)
rem with GCC (w64devkit), plus the release's prebuilt CUDA backend: bin\ggml-cuda.dll of the same commit is loaded at run
rem time by ggml_backend_load_all and binds to this build's ggml-base.dll, so the libraries are named the MSVC way
rem (no "lib" prefix: noprefix.cmake). Output: %QWEN_ROOT%\build\bin
rem Needs gcc (w64devkit), cmake and ninja on PATH, or TOOLS_PATH set to their bin directories (';'-separated).
if not defined QWEN_ROOT set QWEN_ROOT=C:\llama-qwen
if defined TOOLS_PATH set PATH=%TOOLS_PATH%;%PATH%
set HF_UI_VERSION=b11361
cd /d %QWEN_ROOT%
echo === configure %date% %time%
cmake -S src\llama.cpp-a4cb4c61f -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_PROJECT_INCLUDE=%QWEN_ROOT:\=/%/noprefix.cmake || goto fail
echo === build %date% %time%
cmake --build build --target llama-server -j 8 || goto fail
rem (only when missing: a running server keeps them open, and they never change)
for %%f in (ggml-cuda.dll cudart64_12.dll cublas64_12.dll cublasLt64_12.dll) do if not exist build\bin\%%f copy /y %QWEN_ROOT%\bin\%%f build\bin\ >nul || goto fail
echo === BUILD OK %date% %time%
exit /b 0
:fail
echo === BUILD FAILED %date% %time%
exit /b 1
