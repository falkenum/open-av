use babycat::resample::babycat_lanczos::resample;
use cpal::{Data, Sample, SampleFormat, traits::{StreamTrait, DeviceTrait, HostTrait}, Device, Host, SampleRate, StreamInstant, StreamConfig};
use rodio::{source::Repeat, Decoder, Source};
use rustfft::{Fft, FftDirection, num_complex::Complex32, algorithm::Radix4};
use serde::{Serialize, Serializer};
use futures::{stream::{Stream, StreamExt, Buffered, Map, ReadyChunks}, Future, FutureExt, future};
use tokio::{io::{BufReader}};

use std::{io::{Write}, fs::File, collections::VecDeque, sync::MutexGuard, ops::Deref};
// use std::fs::File;
use std::f32::consts::PI;
use std::ops::DerefMut;
use std::sync::{Arc, Mutex};

use crate::NUM_INSTANCES;

const FFT_SIZE: usize = 512;
const WINDOW_SIZE: usize = 256;
const HOP_SIZE: usize = 127;
const MAX_DB: f32 = 60.0;
const MIN_DB: f32 = 10.0;
const MAX_FREQ: usize = 3000;

// 8k items
const SAMPLE_BUFFER_LEN: usize = 1 << 13;

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
    // pub frame_idx: SharedFrameIndex,
    pub processed_data: ProcessedData,
    pub av_data_queue: SharedAvData,
    // pub last_visual_update: Option<StreamInstant>,
    pub next_av_data: Option<AvData>,
    source: Option<AvSource>,

    fft_buffer: [Complex32; FFT_SIZE],
    fft_scratch: [Complex32; FFT_SIZE],
    fft_handle: Radix4<f32>,
}

impl Av {
    pub fn new() -> Self {
        // let (_stream, stream_handle) = OutputStream::try_default().unwrap();

        Self {
            source: None,
            fft_buffer: [Complex32::from(0.0f32); FFT_SIZE],
            fft_scratch: [Complex32::from(0.0f32); FFT_SIZE],
            fft_handle: Radix4::new(FFT_SIZE, FftDirection::Forward),
            processed_data: ProcessedData {
                sample_rate: 0,
                stft_output_db: Vec::new(),
                instance_intensity: Vec::new(),
            },
            av_data_queue: SharedAvData::new(),
            next_av_data: None
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
            stft_output_db: Vec::new(),
            instance_intensity: Vec::new(),
        };

        let mut offset = 0;
        let start_time = std::time::Instant::now();
        while offset < sample_rate as usize * 60 {// 60 seconds of data
            for i in 0..WINDOW_SIZE {
                let sample = match channels {
                    1 => sample_buf[offset + i] as f32 / i16::MAX as f32,
                    2 => (sample_buf[offset + i] as f32 + sample_buf[offset + i + 1] as f32) as f32 / 2.0 / i16::MAX as f32,
                    _ => panic!(),
                };

                let windowed_val = sample * (a0 - (1.0 - a0) * (2.0 * PI * i as f32 / WINDOW_SIZE as f32).cos());
                self.fft_buffer[i] = Complex32::from(windowed_val);
            }

            for i in WINDOW_SIZE..FFT_SIZE {
                self.fft_buffer[i] = Complex32::from(0.0);
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

            processed_data.stft_output_db.push(Vec::from(fft_mag_db));
            processed_data.instance_intensity.push(instance_intensity);
        }
        let out_path = std::env::current_dir().unwrap()
            .as_path().join("res")
            .join("sounds")
            .join(std::path::Path::new(filename).with_extension("pickle"));

        let mut out_fd = std::fs::File::create(out_path).unwrap();

        let pickled_data = serde_pickle::to_vec(&processed_data, Default::default()).unwrap();
        out_fd.write(pickled_data.as_slice()).unwrap();
        out_fd.flush().unwrap();

        self.processed_data = processed_data;

        println!("processing took {} ms ", start_time.elapsed().as_millis());
    }

    pub fn play(&mut self, source_file: &str) {
        let mut source = AvSource::new();
        source.play(source_file, self.av_data_queue.clone());
        self.source = Some(source);
    }
}

// pub type SharedAvData = SharedQueue<AvData>;

// pub struct LockedQueue<'a, T> {
//     inner: MutexGuard<'a, VecDeque<T>>
// }

// impl<'a, T> LockedQueue<'a, T> {
//     pub fn push(&mut self, item: T) {
//         self.inner.deref_mut().push_back(item);
//     }

//     pub fn pop(&mut self) -> Option<T> {
//         self.inner.deref_mut().pop_front()
//     }
// }

// impl<T> Clone for SharedQueue<T> {
//     fn clone(&self) -> Self {
//         Self { inner: Arc::clone(&self.inner) }
//     }
// }

#[derive(Clone)]
pub struct AvData {
    pub frame_index: usize,
    pub callback_time: std::time::Instant,
    pub playback_delay: std::time::Duration,
}

#[derive(Serialize)]
pub struct ProcessedData {
    sample_rate: u32,
    stft_output_db: Vec<Vec<f32>>,
    pub instance_intensity: Vec<[f32; crate::NUM_INSTANCES as usize]>,
}
struct Resampler<T: Stream<Item = f32>> {
    from_rate: u32,
    to_rate: u32,
    channels: u32,
    buffer: VecDeque<f32>,
    source: ReadyChunks<T>,
}

impl<T: Stream<Item = f32> + Unpin> Resampler<T> {
    fn new(input_stream: T, sample_count: usize, from_rate: u32, to_rate: u32, channels: u32) -> Self {
        Self {
            buffer: VecDeque::new(),
            source: input_stream.ready_chunks(sample_count),
            from_rate,
            to_rate,
            channels,
        }
    }
}

impl<T: Stream<Item = f32>> Stream for Resampler<T> where T::Item: future::Future {
    type Item = T::Item;

    fn poll_next(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        if self.buffer.len() == 0 {
            tokio::spawn(async move {
                if let Some(buffer) = self.source.next().await {
                    self.buffer = VecDeque::from(resample(
                        self.from_rate, 
                        self.to_rate, 
                        self.channels, 
                        buffer.as_slice(),
                    ).expect("could not resample"));
                }
            });
        }

    }
}

struct AsyncDecoder {
    inner: Decoder<File>,
}

impl futures::stream::Stream for AsyncDecoder {
    type Item = future::Ready<f32>;

    fn poll_next(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        std::task::Poll::Ready( 
            match self.inner.next() {
                Some(val) => Some(future::ready(val as f32 / i16::MAX as f32)),
                None => None,
            }
        )
    }
}

impl AsyncDecoder {
    fn new(source_file: &str) -> Self {
        AsyncDecoder {
            inner: Decoder::new(File::open(source_file).unwrap()).unwrap(),
        }
    }
}

// impl Deref for AsyncDecoder {
//     type Target = Decoder<File>;

//     fn deref(&self) -> &Self::Target {
//         &self.inner
//     }
// }

// impl futures::Stream for AsyncDecoder {
//     type Item = ;

//     fn poll_next(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
//         todo!()
//     }
// }

impl AvSource {
    pub fn new() -> Self {
        
        let host = cpal::default_host();
        let device = host.default_output_device().expect("no output device available");
        let mut supported_configs_range = device.supported_output_configs()
            .expect("error while querying configs");

        let config = supported_configs_range.next()
            .expect("no supported config?!")
            .with_max_sample_rate()
            .config();


        Self {
            device,
            config,
            output_stream: None,
            // processed_data,
            // resample_count: 1024,
            // av_data_queue,
        }
    }

    pub fn play(&mut self, source_file: &str) {
        let decoder = AsyncDecoder::new(source_file);
        let source_sample_rate = decoder.inner.sample_rate();
        let source_channels = decoder.inner.channels();

        let target_sample_rate = self.config.sample_rate.0;

        // buffer is 1 second of samples
        let raw_samples = decoder.buffered(target_sample_rate);

        let output_samples = Resampler::new(raw_samples, target_sample_rate);

        let sample_idx_update_count =(self.config.sample_rate.0 / (crate::MAX_INSTANCE_UPDATE_RATE_HZ)) as usize;
        let mut current_sample_index = 0;

        let data_callback = move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
            let callback_time = std::time::Instant::now();
            for i in 0..data.len() {
                // if current_sample_index % sample_idx_update_count == 0 {
                //     av_data_queue.push(AvData {
                //         callback_time,
                //         playback_delay: info.timestamp().playback.duration_since(&info.timestamp().callback).unwrap(),
                //         frame_index: current_sample_index / HOP_SIZE,
                //     });
                // }

                // // TODO support when file is mono but device is stereo
                // data[i] = sample_buffer[current_sample_index];

                // current_sample_index += 1;
            }
        };
        let error_callback = |err| {
            panic!("{:?}", err);
        };
        let stream = self.device.build_output_stream(
            &self.config,
            data_callback,
            error_callback,
        ).unwrap();

        stream.play().unwrap();

        self.output_stream = Some(stream);
    }
}
