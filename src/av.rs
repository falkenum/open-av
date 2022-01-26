use rodio::OutputStreamHandle;
use rodio::source::PeriodicAccess;
use rodio::{Decoder, OutputStream, source::Source, source::Repeat};
use rustfft::{Fft, FftDirection, num_complex::Complex32, algorithm::Radix4};

use std::borrow::{BorrowMut, Borrow};
use std::collections::vec_deque::IterMut;
use std::io::BufReader;
use std::fs::File;
use std::f32::consts::PI;
use std::ops::DerefMut;
use std::time::{SystemTime, Duration};
use std::collections::{VecDeque};
use std::vec::Vec;
use std::sync::{Arc, Mutex, RwLock};
use std::boxed::Box;

pub const FFT_SIZE: usize = 1024;
const WINDOW_WIDTH: u32 = 3600;
const WINDOW_HEIGHT: u32 = 400;
const NUM_LEDS: usize = 64;

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
    // profile: Box<dyn AvProfile>,
    stream_handle: OutputStreamHandle,
    _stream: OutputStream,
    pub target_buf: AvBuffer,
}

impl Av {
    pub fn new() -> Self {
        // let mut sample_queue = VecDeque::new();
        // sample_queue.resize(FFT_SIZE, 0);

        let (_stream, stream_handle) = OutputStream::try_default().unwrap();

        Self {
            stream_handle,
            _stream,
            target_buf: Arc::from(Mutex::from([0.0; FFT_SIZE])),
        }
    }
    pub fn play(&self , source: AvSource) {
        self.stream_handle.play_raw(source).unwrap();
    }
}

struct SignalProcessor {
    fft_buffer: [Complex32; FFT_SIZE],
    fft_scratch: [Complex32; FFT_SIZE],
    fft: Radix4<f32>,
}

impl SignalProcessor {
    fn new() -> Self {
        Self {
            fft_buffer: [Complex32::from(0.0f32); FFT_SIZE],
            fft_scratch: [Complex32::from(0.0f32); FFT_SIZE],
            fft: Radix4::new(FFT_SIZE, FftDirection::Forward),
        }
    }

    fn run<B>(&mut self, mut sample_it: B, target_buf: &mut [f32; FFT_SIZE]) where B: Iterator<Item=f32> {
        // assert_eq!(sample_buffer.len(), FFT_SIZE);
        for i in 0..FFT_SIZE {
            let a0 = 25.0 / 46.0;

            let val = sample_it.next().unwrap().clone();
            let windowed_val = val * (a0 - (1.0 - a0) * (2.0 * PI * i as f32 / FFT_SIZE as f32).cos());
            self.fft_buffer[i] = Complex32::from(windowed_val);
        }

        self.fft.process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch);

        for i in 0..FFT_SIZE {
            target_buf[i] = 20.0*self.fft_buffer[i].norm().log10();
        }

        // return &self.output;
    }
}
pub type AvBuffer = Arc<Mutex<[f32; FFT_SIZE]>>;
pub struct AvSource {
    source_buffer: Repeat<Decoder<BufReader<File>>>, 
    sample_queue: VecDeque<f32>,
    sigproc: SignalProcessor,
    last_processed: std::time::Instant,
    target_buf: AvBuffer,
}

impl AvSource {
    pub fn new(source_file: &str, target_buf: AvBuffer) -> Self {
        let source_buffer = Decoder::new(
            BufReader::new(File::open(source_file).unwrap())
        ).unwrap().repeat_infinite();
        let mut sample_queue = VecDeque::new();
        sample_queue.resize(FFT_SIZE, 0.0);
        Self {
            source_buffer,
            sample_queue,
            sigproc: SignalProcessor::new(),
            last_processed: std::time::Instant::now(),
            target_buf,
        }
    }
}


impl Iterator for AvSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let sample = match self.source_buffer.channels() {
            1 => self.source_buffer.next().unwrap() as f32,
            2 => (self.source_buffer.next().unwrap() as f32 + self.source_buffer.next().unwrap() as f32) / 2.0,
            _ => panic!(),
        } / i16::MAX as f32;

        self.sample_queue.pop_front();
        self.sample_queue.push_back(sample);
        if self.last_processed.elapsed() > Duration::from_nanos(1_000_000_000 / 60) {
            let mut lock = self.target_buf.lock().unwrap();
            let target_buf = lock.deref_mut();
            self.sigproc.run(self.sample_queue.iter().cloned(), target_buf);
            // for i in 0..buf.len() {
            //     buf[i] = source_buf[i];
            // }


            self.last_processed = std::time::Instant::now();
        }
        
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
        self.source_buffer.sample_rate()
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}
