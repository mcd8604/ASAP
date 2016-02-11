PROJECT(Camelyon)

INCLUDE_DIRECTORIES(
  .
  ../../
  ${Camelyon_SOURCE_DIR}/../../io/multiresolutionimageinterface
  ${Camelyon_SOURCE_DIR}/../opencv
  ${Boost_INCLUDE_DIRS}
  ${WHOLESLIDEFILTERS_SOURCE_DIR}
  ${DIAGPathology_BINARY_DIR}
)
ADD_DEFINITIONS(${Boost_DEFINITIONS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

SET(Camelyon_src
  Camelyon.cpp
  Slide.cpp
  Segment.cpp
)

SET(Camelyon_headers
  Slide.h
  Segment.h
)

ADD_EXECUTABLE(Camelyon ${Camelyon_src} ${Camelyon_headers})

SET_TARGET_PROPERTIES(Camelyon PROPERTIES DEBUG_POSTFIX _d)
TARGET_LINK_LIBRARIES(Camelyon wholeslidefilters multiresolutionimageinterface)

INSTALL(
  TARGETS Camelyon 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

IF(WIN32)
  SET_TARGET_PROPERTIES(Camelyon  PROPERTIES FOLDER pathology/diag/executables)   
ENDIF(WIN32)
