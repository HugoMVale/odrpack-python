set -xe

PROJECT_DIR="$1"

# this works
# if [ -e /etc/alpine-release ]; then
#     # musllinux (Alpine Linux)
#     echo "Detected musllinux environment. Installing OpenBLAS using apk..."
#     apk add --no-cache openblas-dev
# else
#     # manylinux (CentOS)
#     echo "Detected manylinux environment. Installing OpenBLAS using yum..."
#     yum install -y openblas-devel
# fi

pip install scipy-openblas32
python -c "import scipy_openblas32; print(scipy_openblas32.get_pkg_config())" > $PROJECT_DIR/scipy-openblas.pc
