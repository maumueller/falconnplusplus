file(GLOB SRCFILES
    "*.cpp"
    "*.h"
    "*.c"
    )

add_library(falconn++ ${SRCFILES})
target_link_libraries(falconn++ Eigen3::Eigen)