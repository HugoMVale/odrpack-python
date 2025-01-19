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

# compilers
add_languages('fortran', native: false)
cpp_compiler = meson.get_compiler('cpp')
fortran_compiler = meson.get_compiler('fortran')

# find python
python = import('python').find_installation(pure: false)
python_dep = python.dependency()

# find nanobind (from subprojects)
nanobind_dep = dependency('nanobind', static: true)

# find blas
blas_libs = ['openblas', 'scipy-openblas']
foreach blas_lib : blas_libs
  blas_dep = dependency(blas_lib, required: false)
  if not blas_dep.found()
    blas_dep = cpp_compiler.find_library(blas_lib, required: false)
  endif
  if blas_dep.found()
    break
  endif
endforeach

if blas_dep.found()
  blas_source = []
else
  message('No specific BLAS library detected. Falling back to reference BLAS.')
  blas_source = join_paths(meson.current_source_dir(), 'extern/blas.f')
endif

# buid odrpack95
odrpack95_path = join_paths(meson.current_source_dir(), 'extern/odrpack95')
odrpack95_files = [
  'src/linpack.f',
  'src/blas_interfaces.f90',
  'src/odrpack_kinds.F90',
  'src/odrpack_core.f90',
  'src/odrpack.f90',
  'src/odrpack_reports.f90',
  'c/odrpack_capi.f90'
]

odrpack95_source = []
foreach f : odrpack95_files
  odrpack95_source += files(join_paths(odrpack95_path, f))
endforeach

odrpack95_lib = static_library(
  meson.project_name(),
  sources: [odrpack95_source, blas_source],
  dependencies: [blas_dep],
)

# build extension module
nanobind_ext = python.extension_module(
  '__odrpack',
  'src/odrpack/__odrpack.py.cpp',
  dependencies : [python_dep, nanobind_dep],
  link_with: odrpack95_lib,
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