# examples/server — OpenAI-compatible HTTP server over the dlopen'd
# libvibevoice shared library (python stdlib only, no build step).
#
# Requires VIBEVOICE_SHARED=ON so libvibevoice.dylib/.so exists for the
# server to bind via vv_capi_*.

if(NOT VIBEVOICE_SHARED)
    message(WARNING
        "VIBEVOICE_BUILD_SERVER=ON without VIBEVOICE_SHARED=ON: the server "
        "dlopens libvibevoice, which won't exist. Build with "
        "-DVIBEVOICE_SHARED=ON or ignore if you point --lib elsewhere.")
endif()

# No compilation target — provide convenience vars + a custom target so
# `cmake --build build-shared --target vibevoice-server` works.
set(VIBEVOICE_SERVER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/server.py)

add_custom_target(vibevoice-server
    COMMAND ${CMAKE_COMMAND} -E env
        VIBEVOICE_BACKEND=$ENV{VIBEVOICE_BACKEND}
        ${Python3_EXECUTABLE} ${VIBEVOICE_SERVER_SCRIPT}
        --lib $<TARGET_FILE:vibevoice>
        --tokenizer $ENV{VIBEVOICE_TOKENIZER}
        --host $ENV{VIBEVOICE_HOST}
        --port $ENV{VIBEVOICE_PORT}
    USES_TERMINAL
    COMMENT "Starting vibevoice OpenAI-compatible server"
    VERBATIM)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_EXECUTABLE)
    message(STATUS "vibevoice-server: Python3 not found — target will "
                   "not run until python3 is available")
endif()
