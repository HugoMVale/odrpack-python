set -xe

PROJECT_DIR="$1"

# Temporarily add MSYS2 to PATH
PATH="/c/msys64/usr/bin:$PATH"

# Update pacman database and upgrade system packages
pacman -Sy --noconfirm
pacman -Suu --noconfirm

# Install OpenBLAS
pacman -S --noconfirm mingw-w64-x86_64-openblas


# Check if gcc and gfortran are available
which gcc
which gfortran

