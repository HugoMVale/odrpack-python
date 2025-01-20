set -xe

PROJECT_DIR="$1"

# install OpenBLAS
curl -L -o OpenBLAS-0.3.29_x64.zip https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29_x64.zip
unzip OpenBLAS-0.3.29_x64.zip -d openblas

# add OpenBLAS to PATH
echo "CIBW_ENVIRONMENT_WINDOWS=PATH=%PATH%;$PROJECT_DIR/openblas" >> $env:GITHUB_ENV