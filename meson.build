project('neural_network', 'cpp',
  version: '1.0',
  default_options: ['cpp_std=c++20']
)

exe = executable('neural_network',
  sources: ['src/main.cpp'],
  include_directories: 'include',
  install: true
)

install_headers('include/main.h', subdir: 'neural_network')

