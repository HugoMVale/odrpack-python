set -xe

PROJECT_DIR="$1"

if [ -e /etc/alpine-release ]; then
    # musllinux (Alpine Linux)
    echo "Detected musllinux environment. Installing OpenBLAS using apk..."
    apk add --no-cache openblas-dev
else
    # manylinux (CentOS)
    echo "Detected manylinux environment. Installing OpenBLAS using yum..."
    yum install -y centos-release-scl
    yum install -y devtoolset-12-gcc devtoolset-12-gcc-gfortran devtoolset-12-gcc-c++
    source /opt/rh/devtoolset-12/enable
    export CC=gcc
    export CXX=g++
    export FC=gfortran
    yum install -y openblas-devel
fi
