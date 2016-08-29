# Download eigen
wget http://bitbucket.org/eigen/eigen/get/3.2.9.tar.gz
tar -xzf 3.2.9.tar.gz

# Rename extracted dir as eigen3
mv eigen-eigen-* eigen3

# Set eigen headers into your include path
export CPATH=eigen3/
