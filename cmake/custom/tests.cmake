#.rst:
#
# autocmake.yml configuration::
#
#   docopt: "--enable-tests=<TESTS> Enable tests [default: True]."
#   define: "'-DENABLE_TESTS={0}'.format(arguments['--enable-tests'])"

option(ENABLE_TESTS "Enable test suite" ON)

if(ENABLE_TESTS)
  enable_testing()
  include(CTest)
  add_subdirectory(tests) # This must come last!!
endif()
