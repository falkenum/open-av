// use rodio::OutputStreamHandle;
// use rodio::{Decoder, OutputStream, source::Source, source::Repeat};
use babycat::resample::babycat_lanczos::resample;
use cpal::{Data, Sample, SampleFormat, traits::{StreamTrait, DeviceTrait, HostTrait}, Stream, Device, Host, SampleRate};
use rodio::{source::Repeat, Decoder, Source};
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
    pub processed_data: ProcessedData,
    // host: Host,
    // device: Device,
    // stream: Option<Stream>,
    // source_file: Option<&str>,
    source: Option<AvSource>,

    fft_buffer: [Complex32; FFT_SIZE],
    fft_scratch: [Complex32; FFT_SIZE],
    fft_handle: Radix4<f32>,
}

impl Av {
    pub fn new(sample_idx: SharedFrameIndex) -> Self {
        // let (_stream, stream_handle) = OutputStream::try_default().unwrap();

        Self {
            source: None,
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

    pub fn play(&mut self, shared_frame_index: SharedFrameIndex, source_file: &str) {
        let mut source = AvSource::new(shared_frame_index);
        source.play(source_file);
        self.source = Some(source);
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
    // current_sample_index: usize,
    sample_idx_update_count: usize,
    resample_count: usize,
    shared_frame_index: SharedFrameIndex,
    host: Host,
    device: Device,

    // sample_buffer: Option<Repeat<Decoder<BufReader<File>>>>,
    stream: Option<Stream>,
}

impl AvSource {
    pub fn new(shared_frame_index: SharedFrameIndex) -> Self {
        
        let host = cpal::default_host();
        let device = host.default_output_device().expect("no output device available");

        Self {
            host,
            device,
            stream: None,
            shared_frame_index,
            sample_idx_update_count: HOP_SIZE,
            resample_count: 1024,
            // sample_buffer: None,
        }
    }

    pub fn play(&mut self, source_file: &str) {
        let mut supported_configs_range = self.device.supported_output_configs()
            .expect("error while querying configs");

        let config = supported_configs_range.next()
            .expect("no supported config?!")
            .with_max_sample_rate()
            .config();

        let pre_conversion_sample_buffer = Decoder::new(
            BufReader::new(File::open(source_file).unwrap())
        ).unwrap();
        let sample_rate = pre_conversion_sample_buffer.sample_rate();
        let channels = pre_conversion_sample_buffer.channels();

        // TODO don't load this all at once
        let pre_conversion_sample_buffer: Vec<_> = pre_conversion_sample_buffer.map(|val| val as f32 / i16::MAX as f32).collect();

        let sample_idx_update_count = self.sample_idx_update_count;
        let resample_count = self.resample_count;
        let shared_frame_index = Arc::clone(&self.shared_frame_index);
        let mut current_sample_index = 0;

        let sample_buffer = resample(
            sample_rate, 
            config.sample_rate.0, 
            channels.into(), 
            &pre_conversion_sample_buffer.as_slice(),
// [current_sample_index..(current_sample_index+data.len())]
        ).unwrap();

        let data_callback = move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
            for i in 0..data.len() {
                if current_sample_index % sample_idx_update_count == 0 {
                    *shared_frame_index.lock().unwrap().deref_mut() = current_sample_index / sample_idx_update_count;

                }

                // TODO support when file is mono but device is stereo
                data[i] = sample_buffer[current_sample_index];

                current_sample_index += 1;
            }
        };
        let error_callback = |err| {
            panic!("{:?}", err);
        };
        let stream = self.device.build_output_stream(
            &config,
            data_callback,
            error_callback,
        ).unwrap();

        stream.play().unwrap();

        self.stream = Some(stream);
        // self.sample_buffer = Some(sample_buffer);
    }
}
