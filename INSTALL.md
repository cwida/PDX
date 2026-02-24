
2. *[Optional]* Install [FFTW](https://www.fftw.org/fftw3_doc/Installation-on-Unix.html) (for higher throughput)
```sh
wget https://www.fftw.org/fftw-3.3.10.tar.gz
tar -xvzf fftw-3.3.10.tar.gz
cd fftw-3.3.10
./configure --enable-float --enable-shared  
sudo make
sudo make install
ldconfig
```

export CXX="/usr/bin/clang++-18" # Set proper Clang if needed
