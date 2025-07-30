project('neural_network', 'cpp',
  version: '1.0',
  default_options: ['cpp_std=c++20']
)

src_dir = 'src'
all_sources = run_command('find', src_dir, '-name', '*.cpp', check: true).stdout().split()
main_cpp = src_dir + '/main.cpp'

lib_sources = []
foreach s : all_sources
  if s != main_cpp
    lib_sources += s
  endif
endforeach

eigen_inc = include_directories('/usr/include/eigen3')

tests = executable('neural_network',
  sources: all_sources,
  include_directories: ['include', eigen_inc],
  install: true
)

libnetwork = library('neural_network',
  sources: lib_sources,
  include_directories: ['include', eigen_inc],
  install: true
)

# Optional public headers
install_headers('include/lib.hpp', subdir: 'neural_network')

