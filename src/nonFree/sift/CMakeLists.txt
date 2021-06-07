
# libs should be static
set(BUILD_SHARED_LIBS OFF)

# use PIC code for link into shared lib
if(UNIX)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
endif(UNIX)
set(USE_SSE2 FALSE)
if(NOT USE_SSE2)
  add_definitions(-DVL_DISABLE_SSE2)
endif()

add_definitions(-DVL_DISABLE_THREADS)

include_directories(./vl)
set(FEATS
  vl/aib.c
  vl/array.c
  vl/covdet.c
  vl/dsift.c
  vl/fisher.c
  vl/gmm.c
  vl/generic.c
  vl/getopt_long.c
  vl/hikmeans.c
  vl/hog.c
  vl/homkermap.c
  vl/host.c
  vl/ikmeans.c
  vl/imopv_sse2.c
  vl/imopv.c
  vl/kdtree.c
  vl/kmeans.c
  vl/lbp.c
  vl/liop.c
  vl/mathop_avx.c
  vl/mathop_sse2.c
  vl/mathop.c
  vl/mser.c
  vl/pgm.c
  vl/quickshift.c
  vl/random.c
  vl/sift.c
  )
set_source_files_properties(${FEATS} PROPERTIES LANGUAGE C)
add_library(vlsift ${FEATS})
install(TARGETS vlsift DESTINATION lib EXPORT openMVG-targets)
set_property(TARGET vlsift PROPERTY FOLDER OpenMVG/nonFree)
install(
  DIRECTORY .
  DESTINATION include/openMVG_dependencies/nonFree/sift
  COMPONENT headers
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)
