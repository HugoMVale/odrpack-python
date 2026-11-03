project(
  'odrpack',
  ['cpp', 'fortran'],
  version : '0.7.0',
  meson_version: '>=1.5.0',
  default_options: [
    'buildtype=release',
    'b_ndebug=if-release',
    'cpp_std=c++17',
    'odrpack95:blas_option=refblas'
  ],
)

# python
python = import('python').find_installation(pure: false)
python_dep = python.dependency()

# Free-threaded CPython (3.13t, 3.14t, ...): nanobind -- both the static library
# and the extension -- must be compiled with NB_FREE_THREADED. Otherwise the
# module does not declare Py_MOD_GIL_NOT_USED (CPython silently re-enables the
# GIL on import) and nanobind's internals are not thread-safe. Global arguments
# are used on purpose, because they also reach the nanobind subproject.
if python.get_variable('Py_GIL_DISABLED', 0) == 1
  add_global_arguments('-DNB_FREE_THREADED', language: 'cpp')
endif

# nanobind (from subprojects)
nanobind_dep = dependency('nanobind', static: true)

# odrpack (from subprojects)
odrpack_dep = dependency('odrpack95', static: true)

# build extension module
nanobind_ext = python.extension_module(
'__odrpack',
'src/odrpack/__odrpack.py.cpp',
dependencies : [python_dep, nanobind_dep, odrpack_dep],
install: true,
subdir: 'odrpack'
)

python.install_sources(
  [
  'src/odrpack/__init__.py',
  'src/odrpack/__odrpack.pyi',
  'src/odrpack/odr.py',
  'src/odrpack/exceptions.py',
  'src/odrpack/result.py',
  'src/odrpack/py.typed',
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
