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

# python dependencies
python = import('python').find_installation(pure: false)
python_dep = python.dependency()

# nanobind dependency (from subprojects)
nanobind_dep = dependency('nanobind', static: true)

# blas dependency
# can be made more general to look for accelerate, scipy-openblas, etc.
blas_dep = dependency('openblas', required: false)

if blas_dep.found()
  message('BLAS: ' + blas_dep.name() + ' ' + blas_dep.version())
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