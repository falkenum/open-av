use rodio::OutputStreamHandle;
use rodio::{Decoder, OutputStream, source::Source, source::Repeat};
use rustfft::{Fft, FftDirection, num_complex::Complex32, algorithm::Radix4};
use plotters::prelude::*;
use serde::Serialize;
use serde::ser::SerializeSeq;
use bincode::Options;

use std::io::{BufReader, Write};
use std::fs::File;
use std::f32::consts::PI;
use std::net::TcpStream;
use std::ops::DerefMut;
use std::time::Duration;
use std::sync::{Arc, Mutex};

use crate::NUM_INSTANCES;

const FFT_SIZE: usize = 1024;
const WINDOW_SIZE: usize = 512;
const HOP_SIZE: usize = 256;
const MAX_DB: f32 = 60.0;
const MIN_DB: f32 = 10.0;
const MAX_FREQ: usize = 7000;

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

            let mut fft_mag_db = [0.0f32; FFT_SIZE];
            for i in 0..FFT_SIZE {
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
                instance_intensity[i] = fft_mag_db[dominant_bin_per_instance[i] as usize] / MAX_DB
            }

            // processed_data.sample_buffers.push(sample_buffer);
            processed_data.stft_output_db.push(fft_mag_db);
            processed_data.instance_intensity.push(instance_intensity);
        }
        let out_path = std::env::current_dir().unwrap()
            .as_path().join("res")
            .join("sounds")
            .join(std::path::Path::new(filename).with_extension("bin"));

        // pollster::block_on(processed_data.plot());
        let mut out_fd = std::fs::File::create(out_path).unwrap();
        let bytes = bincode::DefaultOptions::new()
            .with_big_endian().serialize(&processed_data).unwrap();
        out_fd.write(bytes.as_slice()).unwrap();

        self.processed_data = processed_data;

        // let mb = std::mem::size_of::<[f32;FFT_SIZE]>() * processed_data.instance_intensity.len() / pow(2, 20);
        println!("processing took {} ms ", start_time.elapsed().as_millis());
    }
}

pub type SharedFrameIndex = Arc<Mutex<usize>>;

// impl Serialize for [f32; FFT_SIZE] {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer {
//         let mut seq = serializer.serialize_seq(Some(2*FFT_SIZE + crate::NUM_INSTANCES as usize))?;
//         for elt in self.as_slice() {
//             seq.serialize_element(elt)?;
//         }

//         seq.end()
//     }
// }


// #[derive(Serialize)]
pub struct ProcessedData {
    sample_rate: u32,
    // samples: Vec<i16>,
    pub stft_output_db: Vec<[f32; FFT_SIZE]>,
    pub instance_intensity: Vec<[f32; crate::NUM_INSTANCES as usize]>,
}

impl ProcessedData {
    fn write_to_file(&self) {

        // serialize with network byte order
        // let bytes = bincode::DefaultOptions::new()
        //     .with_big_endian().serialize(self).unwrap();
        
        

        // let client = reqwest::blocking::Client::new();
        // let response = client
        //     .post("http://127.0.0.1:8080")
        //     .body(bytes)
        //     .send()
        //     .unwrap();

        // let response = reqwest::blocking::get("http://0.0.0.0:8080")
        // println!("{:?}", response);


        // stream.read(&mut [0; 128])?;
        // Ok(())

        // let root = BitMapBackend::new(path.as_os_str(), (640, 480)).into_drawing_area();
        // root.fill(&WHITE)?;
        // let mut chart = ChartBuilder::on(&root)
        //     // .caption("y=x^2", ("sans-serif", 50).into_font())
        //     .margin(5)
        //     .x_label_area_size(30)
        //     .y_label_area_size(30)
        //     .build_cartesian_2d(-1f32..1f32, -0.1f32..1f32)?;
    
        // chart.configure_mesh().draw()?;

        // let elements = (0..FFT_SIZE).map(|i| { self.sample_rate as f32 / FFT_SIZE as f32 * i as f32 });
    
        // chart
        //     // .draw_series()
        //     .draw_series(LineSeries::new(
        //         (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)),
        //         &RED,
        //     ))?;
        //     // .label("y = x^2")
        //     // .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    
        // chart
        //     .configure_series_labels()
        //     .background_style(&WHITE.mix(0.8))
        //     .border_style(&BLACK)
        //     .draw()?;
    
        // Ok(())
    }
}

impl Serialize for ProcessedData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer {
        serializer.serialize_u32(self.sample_rate)
        // let mut seq = serializer.serialize_seq(Some(1 + FFT_SIZE*self.stft_output_db.len()))?;
        // seq.serialize_element(&self.sample_rate)?;
        // for buf in self.sample_buffers.as_slice() {
        //     for elt in buf {
        //         seq.serialize_element(&elt)?;
        //     }
        // }
        // for buf in self.stft_output_db.as_slice() {
        //     for elt in buf {
        //         seq.serialize_element(elt)?;
        //     }
        // }

        // for buf in self.instance_intensity.as_slice() {
        //     for elt in buf {
        //         seq.serialize_element(elt)?;
        //     }
        // }
        // seq.end()
    }
}

// impl Jsonable for ProcessedData {
//     fn to_json(&self) -> json::JsonValue {
//         json::object! {
//             "sample_buffers": format!("{:?}", self.sample_buffers),
//             "stft_output_db": format!("{:?}", self.stft_output_db),
//             "instance_intensity": format!("{:?}", self.instance_intensity),
//         }
//     }
// }

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

        // self.sample_queue.pop_front();
        // self.sample_queue.push_back(sample);
        if self.current_sample_index % self.sample_idx_update_count == 0 {
            *self.shared_frame_index.lock().unwrap().deref_mut() = self.current_sample_index / self.sample_idx_update_count;
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
