use rodio::OutputStreamHandle;
use rodio::{Decoder, OutputStream, source::Source, source::Repeat};
use rustfft::{Fft, FftDirection, num_complex::Complex32, algorithm::Radix4};
use serde::{Serialize, Serializer};

use std::io::{BufReader, Write};
use std::fs::File;
use std::f32::consts::PI;
use std::ops::DerefMut;
use std::time::Duration;
use std::sync::{Arc, Mutex};

use crate::NUM_INSTANCES;

const FFT_SIZE: usize = 512;
const WINDOW_SIZE: usize = 256;
const HOP_SIZE: usize = 127;
const MAX_DB: f32 = 60.0;
const MIN_DB: f32 = 10.0;
const MAX_FREQ: usize = 3000;

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
//             sample_rate,
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
    pub frame_idx: SharedFrameIndex,
    pub stream_handle: OutputStreamHandle,
    pub processed_data: ProcessedData,
    _stream: OutputStream,

    fft_buffer: [Complex32; FFT_SIZE],
    fft_scratch: [Complex32; FFT_SIZE],
    fft_handle: Radix4<f32>,
}

impl Av {
    pub fn new(sample_idx: SharedFrameIndex) -> Self {
        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        Self {
            stream_handle,
            _stream,
            fft_buffer: [Complex32::from(0.0f32); FFT_SIZE],
            fft_scratch: [Complex32::from(0.0f32); FFT_SIZE],
            fft_handle: Radix4::new(FFT_SIZE, FftDirection::Forward),
            frame_idx: sample_idx,
            processed_data: ProcessedData {
                sample_rate: 0,
                // sample_buffers: Vec::new(),
                stft_output_db: Vec::new(),
                instance_intensity: Vec::new(),
            }
        }
    }

    pub fn process(&mut self, filename: &str) {
        let sample_iter = Decoder::new(
            BufReader::new(File::open(filename).unwrap())
        ).unwrap();

        let channels = sample_iter.channels();
        let sample_rate = sample_iter.sample_rate();
        let sample_buf: Vec<i16> = sample_iter.collect();
        let num_samples = sample_buf.len();

        let a0 = 25.0 / 46.0;
        let mut processed_data = ProcessedData {
            sample_rate,
            // sample_buffers: Vec::new(),
            stft_output_db: Vec::new(),
            instance_intensity: Vec::new(),
        };

        // processed_data.plot();

        let mut offset = 0;
        let start_time = std::time::Instant::now();
        while offset < sample_rate as usize * 60 {// 60 seconds of data
            // let mut sample_buffer = [0.0; FFT_SIZE];
            for i in 0..WINDOW_SIZE {
                // let sample_idx = i / channels as usize;
                let sample = match channels {
                    1 => sample_buf[offset + i] as f32 / i16::MAX as f32,
                    2 => (sample_buf[offset + i] as f32 + sample_buf[offset + i + 1] as f32) as f32 / 2.0 / i16::MAX as f32,
                    _ => panic!(),
                };

                let windowed_val = sample * (a0 - (1.0 - a0) * (2.0 * PI * i as f32 / WINDOW_SIZE as f32).cos());
                self.fft_buffer[i] = Complex32::from(windowed_val);

                // sample_buffer[i] = windowed_val;
            }

            for i in WINDOW_SIZE..FFT_SIZE {
                self.fft_buffer[i] = Complex32::from(0.0);
                // sample_buffer[i] = 0.0;
            }

            self.fft_handle.process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch);

            let mut fft_mag_db = [0.0f32; FFT_SIZE/2];
            for i in 0..(FFT_SIZE/2) {
                fft_mag_db[i] = 20.0 * self.fft_buffer[i].norm().log10();
                if fft_mag_db[i] > MAX_DB {
                    fft_mag_db[i] = MAX_DB;
                }
            }

            offset += HOP_SIZE;

            let max_bin = FFT_SIZE * MAX_FREQ / sample_rate as usize;
            let bins_per_instance = max_bin as f32 / crate::NUM_INSTANCES as f32;

            let mut dominant_bin_per_instance = [-1i32; crate::NUM_INSTANCES as usize];

            for i in 0..max_bin {
                let instance_num = (i as f32 / max_bin as f32 * crate::NUM_INSTANCES as f32) as usize;

                if dominant_bin_per_instance[instance_num] == -1 || fft_mag_db[i] > fft_mag_db[dominant_bin_per_instance[instance_num] as usize] {
                    dominant_bin_per_instance[instance_num] = i as i32;
                }
            }

            let mut instance_intensity = [0.0f32; crate::NUM_INSTANCES as usize];
            for i in 0..NUM_INSTANCES as usize {
                instance_intensity[i] = fft_mag_db[dominant_bin_per_instance[i] as usize] / MAX_DB;

                if instance_intensity[i] < 0.0 {
                    instance_intensity[i] = 0.0;
                }
            }

            // processed_data.sample_buffers.push(sample_buffer);
            processed_data.stft_output_db.push(fft_mag_db);
            processed_data.instance_intensity.push(instance_intensity);
        }
        let out_path = std::env::current_dir().unwrap()
            .as_path().join("res")
            .join("sounds")
            .join(std::path::Path::new(filename).with_extension("pickle"));

        let mut out_fd = std::fs::File::create(out_path).unwrap();
        let serializable_data = processed_data.to_serializable();

        let pickled_data = serde_pickle::to_vec(&serializable_data, Default::default()).unwrap();
        out_fd.write(pickled_data.as_slice()).unwrap();
        out_fd.flush().unwrap();

        self.processed_data = processed_data;

        println!("processing took {} ms ", start_time.elapsed().as_millis());
    }
}

pub type SharedFrameIndex = Arc<Mutex<usize>>;

pub struct ProcessedData {
    sample_rate: u32,
    pub stft_output_db: Vec<[f32; FFT_SIZE / 2]>,
    pub instance_intensity: Vec<[f32; crate::NUM_INSTANCES as usize]>,
}

impl ProcessedData {
    fn to_serializable(&self) -> SerializableData {
        SerializableData {
            sample_rate: self.sample_rate,
            stft_output_db: self.stft_output_db.iter().map(|v| Vec::from(*v)).collect()
        }
    }
}

#[derive(Serialize)]
struct SerializableData {
    sample_rate: u32,
    stft_output_db: Vec<Vec<f32>>,
}

pub struct AvSource {
    current_sample_index: usize,
    sample_idx_update_count: usize,
    shared_frame_index: SharedFrameIndex,
    sample_buffer: Repeat<Decoder<BufReader<File>>>,
}

impl AvSource {
    pub fn new(filename: &str, shared_sample_index: SharedFrameIndex) -> Self {
        let sample_buffer = Decoder::new(
            BufReader::new(File::open(filename).unwrap())
        ).unwrap().repeat_infinite();

        Self {
            shared_frame_index: shared_sample_index,
            sample_idx_update_count: HOP_SIZE,
            current_sample_index: 0,
            sample_buffer,
        }
    }
}


impl Iterator for AvSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {

        if self.current_sample_index % self.sample_idx_update_count == 0 {
            *self.shared_frame_index.lock().unwrap().deref_mut() = self.current_sample_index / self.sample_idx_update_count;
        }
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
