# ARDFTSRC
ARDFTSRC Audio Resampler, by mycroft, port to C and Windows by dBpoweramp.com

Usage: pass on command line : "c:\in.wav" "c:\out.wav" 48000 (out frequency) 16 (out bitdepth) 2048 [optional quality] 0.95 [optional bandwidth]

Quality should be high if increasing bandwidth, example ardftsrc.exe "c:\in.wav" "c:\out.wav" 48000 24 8192 0.99

Requirements (add to project folder):  

                                      \AudioFile\     (simple audio read and write)

                                       \Eigen\         (matrix/vector)

                                       \fftw3\          (FFT operations   also lib file and dll fftw\x64\libfftw3-3.lib)
