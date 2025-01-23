set -xe

PROJECT_DIR="$1"

# Install pkgconfiglite
choco install -y --no-progress --stoponfirstfailure pkgconfiglite
which pkg-config

# Update pacman database and upgrade system packages
PATH="/c/msys64/usr/bin:$PATH"
pacman -Sy --noconfirm
pacman -Suu --noconfirm

# Install OpenBLAS and pkg-config
pacman -S --noconfirm mingw-w64-x86_64-openblas
# pacman -S --noconfirm mingw-w64-x86_64-pkg-config

# Set PKG_CONFIG_PATH for OpenBLAS
ls -l /mingw64/lib/pkgconfig
cp /mingw64/lib/pkgconfig/openblas.pc /c/ProgramData/Chocolatey/lib/pkgconfiglite/tools/pkg-config-lite-0.28-1/lib/pkgconfig/

# Convert MSYS2 path to Unix-style (in case of GitHub Actions path issues)
# export PKG_CONFIG_PATH=$(cygpath -u "C:/msys64/mingw64/lib/pkgconfig")
# echo "PKG_CONFIG_PATH is set to: $PKG_CONFIG_PATH"

# Verify OpenBLAS detection
pkg-config --version
pkg-config --variable=pc_path pkg-config
pkg-config --modversion openblas
