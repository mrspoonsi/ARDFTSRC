// Ardftsrc.cpp : ARDFTSRC - Licensed under WTFPL\n  Created by mycroft @ hydrogenaudio.org\n  C port by the team @ dBpoweramp.com
//


#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <numeric>
#include <complex>
#include "AudioFile.h" // For WAV file I/O
#include <Eigen/Dense>  // For matrix/vector operations
#include <fftw3.h>      // For FFT operations

// Helper function to convert AudioFile's buffer to an Eigen Matrix
Eigen::MatrixXd audioBufferToEigenMatrix(const AudioFile<double>::AudioBuffer& buffer) {
    size_t numChannels = buffer.size();
    size_t numSamples = (numChannels > 0) ? buffer[0].size() : 0;
    Eigen::MatrixXd matrix(numSamples, numChannels);
    for (int channel = 0; channel < numChannels; ++channel) {
        for (int sample = 0; sample < numSamples; ++sample) {
            matrix(sample, channel) = buffer[channel][sample];
        }
    }
    return matrix;
}

// Helper function to convert an Eigen Matrix back to AudioFile's buffer
AudioFile<double>::AudioBuffer eigenMatrixToAudioBuffer(const std::vector< Eigen::MatrixXd >& lstMatrix) {
    assert(!lstMatrix.empty());
    Eigen::Index numSamples = 0, numChannels = 0;
    for (auto& matrix : lstMatrix) {
        numSamples += matrix.rows();
        assert(numChannels == 0 || numChannels == matrix.cols());
        numChannels = matrix.cols();
    }
    AudioFile<double>::AudioBuffer buffer(numChannels, std::vector<double>(numSamples));
    Eigen::Index samplesDone = 0;
    for (auto& matrix : lstMatrix) {
        const Eigen::Index numSamplesHere = matrix.rows();
        for (int channel = 0; channel < numChannels; ++channel) {
            for (int sample = 0; sample < numSamplesHere; ++sample) {
                buffer[channel][samplesDone + sample] = matrix(sample, channel);
            }
        }
        samplesDone += numSamplesHere;
    }
    return buffer;
}

int main(int argc, char* argv[])
{
    // ######################################

    std::cout << ("ARDFTSRC - Licensed under WTFPL\n  Created by mycroft @ hydrogenaudio.org\n  C port by the team @ dBpoweramp.com\n");
    if (argc < 5)
    {
        std::cout << "\nUsage pass on command line : \"c:\\in.wav\" \"c:\\out.wav\" 48000 (out frequency) 16 (out bitdepth) 2048 [optional quality] 0.95 [optional bandwidth]\n\n  Quality should be high if increasing bandwidth, example ardftsrc.exe \"c:\\in.wav\" \"c:\\out.wav\" 48000 24 8192 0.99";
        return(0);
    }

    const std::string input_file = argv[1];
    const std::string output_file = argv[2];
    const int output_samplerate = strtol(argv[3], NULL, 10);
    const int output_bitdepth = strtol(argv[4], NULL, 10);
    int quality = 2048;
    double bandwidth = 0.95;
    if (argc >= 7)
    {
        quality = strtol(argv[5], NULL, 10);
        bandwidth = strtod(argv[6], NULL);
    }
    // ######################################

    AudioFile<double> inputAudioFile;
    if (!inputAudioFile.load(input_file)) {
        std::cerr << "Error: Could not load input file " << input_file << std::endl;
        return 1;
    }
    const int in_samplerate = inputAudioFile.getSampleRate();
    const int in_samplebitdepth = inputAudioFile.getBitDepth();

    std::cout << "\nSource: ";
    std::cout << input_file;
    std::cout << "\n  Sample Rate:";
    std::cout << in_samplerate;
    std::cout << "\n  Bit Depth:";
    std::cout << in_samplebitdepth;
    std::cout << "\n  Channels:";
    std::cout << inputAudioFile.getNumChannels();

    std::cout << "\n\nOutput: ";
    std::cout << output_file;
    std::cout << "\n  Sample Rate:";
    std::cout << output_samplerate;
    std::cout << "\n  Bit Depth:";
    std::cout << output_bitdepth;
    std::cout << "\n  Channels:";
    std::cout << inputAudioFile.getNumChannels();
    std::cout << "\n\nParameters  Quality: ";
    std::cout << quality;
    std::cout << "   Bandwidth: ";
    std::cout << bandwidth;
    std::cout << "\n\n";

    int input_samplerate = inputAudioFile.getSampleRate();
    Eigen::MatrixXd x = audioBufferToEigenMatrix(inputAudioFile.samples);
    int num_channels = (int)x.cols();

    int common_divisor = std::gcd(input_samplerate, output_samplerate);
    long in_nb_samples = input_samplerate / common_divisor;
    long out_nb_samples = output_samplerate / common_divisor;

    long factor = static_cast<long>(2.0 * std::ceil(quality / (2.0 * out_nb_samples)));
    in_nb_samples *= factor;
    out_nb_samples *= factor;

    long in_rdft_size = in_nb_samples * 2;
    long out_rdft_size = out_nb_samples * 2;
    long in_offset = (in_rdft_size - in_nb_samples) / 2;
    long out_offset = (out_rdft_size - out_nb_samples) / 2; // This is not used in the original logic but kept for consistency
    long tr_nb_samples = std::min(in_nb_samples, out_nb_samples);
    long taper_samples = static_cast<long>(tr_nb_samples * (1.0 - bandwidth));

    long size = (long)x.rows();
    long pad_size = size % in_nb_samples;
    if (pad_size > 0) {
        pad_size = in_nb_samples - pad_size;
        Eigen::MatrixXd padding = Eigen::MatrixXd::Zero(pad_size, num_channels);
        Eigen::MatrixXd x_padded(x.rows() + padding.rows(), x.cols());
        x_padded << x, padding;
        x = x_padded;
    }

    int num_chunks = (int)(x.rows() / in_nb_samples);
    // Eigen::MatrixXd y(0, num_channels);
    std::vector< Eigen::MatrixXd > y; y.reserve(num_chunks);

    Eigen::MatrixXd prev_chunk = Eigen::MatrixXd::Zero(out_nb_samples, num_channels);
    Eigen::VectorXcd taper = Eigen::VectorXcd::Zero(in_rdft_size / 2 + 1);

    for (int idx = 0; idx < taper.size(); ++idx) {
        if (idx < tr_nb_samples - taper_samples) {
            taper(idx) = 1.0;
        }
        else if (idx < tr_nb_samples - 1) {
            double n = idx - (tr_nb_samples - taper_samples);
            double t = taper_samples;
            double zbk = t / ((t - n) - 1.0) - t / (n + 1.0);
            double v = 1.0 / (std::exp(zbk) + 1.0);
            taper(idx) = v;
        }
        else {
            taper(idx) = 0.0;
        }
    }

    // These types have identical binary layout, we use them interchangably, to avoid issues with fftw_complex in a vector.
    static_assert(sizeof(fftw_complex) == sizeof(std::complex<double>));

    // FFTW setup
    std::vector<double> fftw_in(in_rdft_size);
    std::vector<std::complex<double> > fftw_out((in_rdft_size / 2) + 1);
    fftw_plan rfft_plan = fftw_plan_dft_r2c_1d(in_rdft_size, fftw_in.data(), reinterpret_cast<fftw_complex*>(fftw_out.data()), FFTW_ESTIMATE);

    std::vector<std::complex<double> > ifftw_in(out_rdft_size / 2 + 1);
    std::vector<double> ifftw_out(out_rdft_size);
    fftw_plan irfft_plan = fftw_plan_dft_c2r_1d(out_rdft_size, reinterpret_cast<fftw_complex*>(ifftw_in.data()), ifftw_out.data(), FFTW_ESTIMATE);

    const double scale = (double)out_nb_samples / (double)in_nb_samples;

    for (int i = 0; i < num_chunks; ++i) {
        std::cout << "Resampling " << std::fixed << std::setprecision(2) << (100.0 * i / num_chunks) << " %\r" << std::flush;
        Eigen::MatrixXd x_chunk = x.block(i * in_nb_samples, 0, in_nb_samples, num_channels);

        Eigen::MatrixXd processed_channels(out_nb_samples * 2, num_channels);

        for (int ch = 0; ch < num_channels; ++ch) {
            // Pad chunk
            std::fill(fftw_in.begin(), fftw_in.end(), 0.0);
            for (int k = 0; k < in_nb_samples; ++k) {
                fftw_in[k + in_offset] = x_chunk(k, ch);
            }
            // RFFT
            fftw_execute(rfft_plan);

            // Apply taper and handle size change
            std::fill(ifftw_in.begin(), ifftw_in.end(), 0.0);
            long freq_domain_size_to_copy = std::min((long)ifftw_in.size(), (long)fftw_out.size());
            for (int k = 0; k < freq_domain_size_to_copy; ++k) {
                ifftw_in[k] = fftw_out[k] * taper(k);

            }

            // IRFFT
            fftw_execute(irfft_plan);

            // Normalize and store
            for (int k = 0; k < out_nb_samples * 2; ++k) {
                processed_channels(k, ch) = ifftw_out[k] / out_rdft_size;
            }
        }

        Eigen::MatrixXd current_chunk = processed_channels.topRows(out_nb_samples);
        current_chunk += prev_chunk;
        current_chunk *= scale;

        y.emplace_back(std::move(current_chunk));

        prev_chunk = processed_channels.bottomRows(out_nb_samples);
    }

    std::cout << "Resampling 100.00 %" << std::endl;

    // Clean up FFTW plans
    fftw_destroy_plan(rfft_plan);
    fftw_destroy_plan(irfft_plan);

    if (y.empty()) {
        std::cerr << "No output was generated" << std::endl;
        return 1;
    }

    AudioFile<double> outputAudioFile;
    outputAudioFile.setSampleRate(output_samplerate);
    outputAudioFile.setAudioBuffer(eigenMatrixToAudioBuffer(y));
    outputAudioFile.setBitDepth(output_bitdepth);
    if (!outputAudioFile.save(output_file, AudioFileFormat::Wave)) {
        std::cerr << "Error: Could not save output file " << output_file << std::endl;
        return 1;
    }

    return 0;
}