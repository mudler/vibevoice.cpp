add_subdirectory(cli)
add_subdirectory(quantize)

if(VIBEVOICE_BUILD_SERVER)
    add_subdirectory(server)
endif()
