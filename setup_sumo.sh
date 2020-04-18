# sumo==1.1.0
brew tap cts198859/sumo
brew cask install xquartz
brew install sumo
# replace libproj.19.dylib with whatever lib version you have for brew
ln -s /usr/local/opt/proj/lib/libproj.19.dylib /usr/local/opt/proj/lib/libproj.13.dylib
export SUMO_HOME=/usr/local/opt/sumo/share/sumo
export PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH