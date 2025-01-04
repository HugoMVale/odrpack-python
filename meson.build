project(
  'odrpack',
  ['fortran', 'c', 'cpp'],
  version : '0.0.1',
  meson_version: '>=1.5.0',
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'cpp_std=c++17',
    'c_std=c99'
  ],
)

# compilers
cpp = meson.get_compiler('cpp')
ff = meson.get_compiler('fortran')

# python and dependencies
python = import('python').find_installation(pure: false)
python_dep = python.dependency()
# openblas_dep = dependency('openblas', required: true)
pybind11_dep = dependency('pybind11', required: true)

# buid odrpack fortran
odrpack_path = join_paths(meson.current_source_dir(), 'extern/odrpack95')
odrpack_files = [
  'src/linpack.f',
  'src/blas_interfaces.f90',
  'src/odrpack_kinds.F90',
  'src/odrpack_core.f90',
  'src/odrpack.f90',
  'src/odrpack_reports.f90',
  'c/odrpack_capi.f90'
]

odrpack_source = []
foreach f : odrpack_files
  odrpack_source += files(join_paths(odrpack_path, f))
endforeach

blas_source = join_paths(meson.current_source_dir(), 'extern/blas.f')

odrpack_lib = static_library(
  meson.project_name(),
  sources: [odrpack_source, blas_source],
  # dependencies: [openblas_dep],
)

# build extension module
pybind_ext = python.extension_module(
 '__odrpack',
'src/odrpack/__odrpack.py.cpp',
 dependencies : [python_dep, pybind11_dep],
 link_with: odrpack_lib,
 include_directories: include_directories('extern/odrpack95/c/include/odrpack'),
 install : true,
 subdir: 'odrpack'
)

python.install_sources(
  [
  'src/odrpack/__init__.py',
  'src/odrpack/driver.py',
  'src/odrpack/exceptions.py',
  'src/odrpack/result.py',
  'src/odrpack/py.typed',
  'src/odrpack/__odrpack.pyi',
  ],
  subdir: 'odrpack'
)

# Generate stubs
# python -m pybind11_stubgen odrpack.__odrpack --output-dir "src"
# stubgen_command = [
#     python,
#     '-m', 'pybind11_stubgen',
#     'odrpack.__odrpack',
#     '--output-dir', 'src'
# ]

# stubgen_target = custom_target(
#     'generate-stubs',
#     command: stubgen_command,
#     input: [pybind_ext],
#     output: 'stubs',
#     depends: [pybind_ext],
#     build_by_default: true
# )