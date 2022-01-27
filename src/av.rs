use lazy_static::__Deref;
use rodio::OutputStreamHandle;
use rodio::{Decoder, OutputStream, source::Source, source::Repeat};
use rustfft::{Fft, FftDirection, num_complex::Complex32, algorithm::Radix4};

use core::num;
use std::io::BufReader;
use std::fs::File;
use std::f32::consts::PI;
use std::ops::DerefMut;
use std::time::Duration;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

const FFT_SIZE: usize = 1024;
const HOP_SIZE: usize = 256;
const MAX_DB: f32 = 60.0;
const MAX_FREQ: usize = 5000;

// trait AvProfile {
//     fn set_led_colors(&mut self, led_colors: &mut [Color], fft_mag_db: &[f32]);
// }

// struct Blue {
//     dominant_bin_per_led: [i32; NUM_LEDS],
//     sample_rate: usize,
// }

// impl Blue {
//     fn new(sample_rate: usize) -> Blue {
//         Blue {
//             dominant_bin_per_led: [-1i32; NUM_LEDS],
//             sample_rate: sample_rate,
//         }
//     }
// }

// impl AvProfile for Blue {
//     fn set_led_colors(&mut self, led_colors: &mut [Color], fft_mag_db: &[f32]) {
//         let max_bin = FFT_SIZE * MAX_FREQ / self.sample_rate as usize;
//         let bins_per_led = max_bin as f32 / NUM_LEDS as f32;

//         self.dominant_bin_per_led = [-1i32; NUM_LEDS];

//         for i in 0..max_bin {
//             let led_num = (i as f32 / bins_per_led) as usize;

//             if self.dominant_bin_per_led[led_num] == -1 || fft_mag_db[i] > fft_mag_db[self.dominant_bin_per_led[led_num] as usize] {
//                 self.dominant_bin_per_led[led_num] = i as i32;
//             }
//         }

//         for i in 0..NUM_LEDS {
//             led_colors[i] = Color::rgba(0, 0, 255, (fft_mag_db[self.dominant_bin_per_led[i] as usize] / MAX_DB * 255.0) as u8)
//         }
//     }
// }

pub struct Av {
    pub sample_idx: SharedSampleIndex,
    pub stream_handle: OutputStreamHandle,
    pub processed_data: ProcessedData,
    _stream: OutputStream,

    fft_buffer: [Complex32; FFT_SIZE],
    fft_scratch: [Complex32; FFT_SIZE],
    fft_handle: Radix4<f32>,
}

impl Av {
    pub fn new(sample_idx: SharedSampleIndex) -> Self {
        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        Self {
            stream_handle,
            _stream,
            fft_buffer: [Complex32::from(0.0f32); FFT_SIZE],
            fft_scratch: [Complex32::from(0.0f32); FFT_SIZE],
            fft_handle: Radix4::new(FFT_SIZE, FftDirection::Forward),
            sample_idx,
            processed_data: ProcessedData {
                stft_output_db: Vec::new(),
            }
        }
    }

    pub fn process(&mut self, filename: &str) {
        let sample_iter = Decoder::new(
            BufReader::new(File::open(filename).unwrap())
        ).unwrap();

        let channels = sample_iter.channels();
        let sample_buf: Vec<i16> = sample_iter.collect();
        let num_samples = sample_buf.len();

        let a0 = 25.0 / 46.0;
        let mut stft_output_db: Vec<[f32; FFT_SIZE]> = Vec::new();

        let mut offset = 0;
        while offset < (num_samples / channels as usize - FFT_SIZE) {
            for i in 0..FFT_SIZE {
                // let sample_idx = i / channels as usize;
                let sample = match channels {
                    1 => sample_buf[offset + i] as f32 / i16::MAX as f32,
                    2 => (sample_buf[offset + i] as f32 + sample_buf[offset + i + 1] as f32) as f32 / 2.0 / i16::MAX as f32,
                    _ => panic!(),
                };

                let windowed_val = sample * (a0 - (1.0 - a0) * (2.0 * PI * i as f32 / FFT_SIZE as f32).cos());
                self.fft_buffer[i] = Complex32::from(windowed_val);
            }

            self.fft_handle.process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch);

            let mut result = [0.0f32; FFT_SIZE];
            for i in 0..FFT_SIZE {
                result[i] = self.fft_buffer[i].norm();
            }
            stft_output_db.push(result);

            offset += HOP_SIZE;
        }
        self.processed_data = ProcessedData {
            stft_output_db
        };
    }
}

pub type SharedSampleIndex = Arc<Mutex<usize>>;

pub struct ProcessedData {
    stft_output_db: Vec<[f32; FFT_SIZE]>
}

pub struct AvSource {
    current_sample_index: usize,
    sample_idx_update_count: usize,
    shared_sample_index: SharedSampleIndex,
    sample_buffer: Repeat<Decoder<BufReader<File>>>,
}

impl AvSource {
    pub fn new(filename: &str, shared_sample_index: SharedSampleIndex) -> Self {
        let sample_buffer = Decoder::new(
            BufReader::new(File::open(filename).unwrap())
        ).unwrap().repeat_infinite();

        Self {
            shared_sample_index,
            sample_idx_update_count: 400,
            current_sample_index: 0,
            sample_buffer,
        }
    }
}


impl Iterator for AvSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {

        // self.sample_queue.pop_front();
        // self.sample_queue.push_back(sample);
        if self.current_sample_index % self.sample_idx_update_count == 0 {
            *self.shared_sample_index.lock().unwrap().deref_mut() = self.current_sample_index;
        }
        // self.sigproc.run(self.sample_queue.iter().cloned());
        
        self.current_sample_index += 1;

        let sample = match self.sample_buffer.channels() {
            1 => self.sample_buffer.next().unwrap() as f32,
            2 => (self.sample_buffer.next().unwrap() as f32 + self.sample_buffer.next().unwrap() as f32) / 2.0,
            _ => panic!(),
        } / i16::MAX as f32;

        Some(sample)
    }
}

impl Source for AvSource {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_buffer.sample_rate()
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}
