set -xe

PROJECT_DIR="$1"

# Check if gcc and gfortran are available
which gcc
which gfortran
gcc --version
gfortran --version

# Update pacman database and upgrade system packages
pacman -Sy --noconfirm
pacman -Suu --noconfirm

# Install OpenBLAS and pkg-config
pacman -S --noconfirm mingw-w64-x86_64-openblas
pacman -S --noconfirm mingw-w64-x86_64-pkg-config

# Set PKG_CONFIG_PATH for OpenBLAS
ls -l /mingw64/lib/pkgconfig

# Convert MSYS2 path to Unix-style (in case of GitHub Actions path issues)
export PKG_CONFIG_PATH=$(cygpath -u "C:/msys64/mingw64/lib/pkgconfig")

# Debugging: Check what the PKG_CONFIG_PATH is set to
echo "PKG_CONFIG_PATH is set to: $PKG_CONFIG_PATH"

# Verify OpenBLAS detection
pkg-config --modversion openblas
