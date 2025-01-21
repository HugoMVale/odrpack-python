set -xe

PROJECT_DIR="$1"

# install OpenBLAS
curl -L -o OpenBLAS-0.3.29_x64.zip https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29_x64.zip
unzip OpenBLAS-0.3.29_x64.zip -d openblas
pwd
ls -l

# install pkgconfiglite, because the image constains a broken version
choco install -y --no-progress --stoponfirstfailure --checksum 6004DF17818F5A6DBF19CB335CC92702 pkgconfiglite

echo $PKG_CONFIG_PATH
pkg-config --variable pc_path pkg-config

# Use PROJECT_DIR to set the paths correctly relative to the project root
# export LIBRARY_PATH="$PROJECT_DIR/openblas/lib"
# export INCLUDE="$PROJECT_DIR/openblas/include"
# export PKG_CONFIG_PATH="$PROJECT_DIR/openblas/lib/pkgconfig"

# # Set the PATH to include OpenBLAS bin directory
# echo "CIBW_ENVIRONMENT_WINDOWS=LIBRARY_PATH=$LIBRARY_PATH INCLUDE=$INCLUDE PATH=%PATH%;$PROJECT_DIR/openblas/bin"
