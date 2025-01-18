project(
  'odrpack',
  ['cpp'],
  version : '0.1.0',
  meson_version: '>=1.5.0',
  default_options: [
    'buildtype=release',
    'b_ndebug=if-release',
    'cpp_std=c++17',
  ],
)

# Compilers
add_languages('fortran', native: false)

# compilers
cpp = meson.get_compiler('cpp')
ff = meson.get_compiler('fortran')

# python and dependencies
python = import('python').find_installation(pure: false)
python_dep = python.dependency()
nanobind_dep = dependency('nanobind', static: true)
# openblas_dep = dependency('openblas', required: true)

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
nanobind_ext = python.extension_module(
 '__odrpack',
'src/odrpack/__odrpack.py.cpp',
 dependencies : [python_dep, nanobind_dep],
 link_with: odrpack_lib,
 include_directories: include_directories('extern/odrpack95/c/include'),
 install: true,
 # limited_api: '3.12',
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
# python -m nanobind.stubgen -m odrpack.__odrpack --output-dir "src/odrpack"
# stubgen_command = [
#     python,
#     '-m', 'nanobind.stubgen',
#     'odrpack.__odrpack',
#     '--output-dir', 'src/odrpack'
# ]

# stubgen_target = custom_target(
#     'generate-stubs',
#     command: stubgen_command,
#     input: [nanobind_ext],
#     output: 'stubs',
#     depends: [nanobind_ext],
#     build_by_default: true
# )