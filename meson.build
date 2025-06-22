project(
  'odrpack',
  ['cpp'],
  version : '0.3.0',
  meson_version: '>=1.5.0',
  default_options: [
    'buildtype=release',
    'b_ndebug=if-release',
    'cpp_std=c++17'
  ],
)

# compilers
add_languages('fortran', native: false)

# python
python = import('python').find_installation(pure: false)
python_dep = python.dependency()

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
  # limited_api: '3.12',
  subdir: 'odrpack'
)

python.install_sources(
  [
  'src/odrpack/__init__.py',
  'src/odrpack/odrpack_style.py',
  'src/odrpack/scipy_style.py',
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
