set -xe

PROJECT_DIR="$1"

if [ -e /etc/alpine-release ]; then
    # musllinux (Alpine Linux)
    echo "Detected musllinux environment. Installing OpenBLAS using apk..."
    apk add --no-cache openblas-dev
else
    # manylinux (CentOS)
    echo "Detected manylinux environment. Installing OpenBLAS and GCC-13 using yum..."
    yum install -y centos-release-scl
    yum install -y devtoolset-13
    yum install -y openblas-devel
    scl enable devtoolset-13 bash
fi

