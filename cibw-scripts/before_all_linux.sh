set -xe

PROJECT_DIR="$1"

if [ -e /etc/alpine-release ]; then
    # musllinux (Alpine Linux)
    echo "Detected musllinux environment. Installing OpenBLAS using apk..."
    apk add --no-cache openblas-dev
else
    # manylinux (CentOS)
    echo "Detected manylinux environment. Installing OpenBLAS using yum..."
    yum install -y openblas-devel
fi
