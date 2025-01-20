set -xe

PROJECT_DIR="$1"

# install OpenBLAS

# pip install scipy-openblas32
# python -c "import scipy_openblas32; print(scipy_openblas32.get_pkg_config())" > $PROJECT_DIR/scipy-openblas.pc

# install delvewheel to repare wheels
pip install delvewheel