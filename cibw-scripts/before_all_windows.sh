set -xe

PROJECT_DIR="$1"

# install OpenBLAS
curl -L -o OpenBLAS-0.3.29_x64.zip https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29_x64.zip
unzip OpenBLAS-0.3.29_x64.zip -d openblas

# Use PROJECT_DIR to set the paths correctly relative to the project root
export LIBRARY_PATH="$PROJECT_DIR/openblas/lib"
export INCLUDE="$PROJECT_DIR/openblas/include"
export PKG_CONFIG_PATH="$PROJECT_DIR/openblas/lib/pkgconfig"

# Set the PATH to include OpenBLAS bin directory
echo "CIBW_ENVIRONMENT_WINDOWS=LIBRARY_PATH=$LIBRARY_PATH INCLUDE=$INCLUDE PATH=%PATH%;$PROJECT_DIR/openblas/bin"
