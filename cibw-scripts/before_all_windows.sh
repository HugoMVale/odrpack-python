set -xe

PROJECT_DIR="$1"

# Check if gcc and gfortran are available
which gcc
which gfortran
gcc --version
gfortran --version

# Temporarily add MSYS2 to PATH
PATH="/c/msys64/usr/bin:$PATH"

# Update pacman database and upgrade system packages
pacman -Sy --noconfirm
pacman -Suu --noconfirm

# Install OpenBLAS and pkg-config
pacman -S --noconfirm mingw-w64-x86_64-openblas
pacman -S --noconfirm mingw-w64-x86_64-pkg-config

# Add only /mingw64/bin to PATH
export PATH="/mingw64/bin:$PATH"

# Verify OpenBLAS detection
pkg-config --modversion openblas