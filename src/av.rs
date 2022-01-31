
use babycat::resample::babycat_lanczos::resample;
use cpal::{Data, Sample, SampleFormat, traits::{StreamTrait, DeviceTrait, HostTrait}, Device, Host, SampleRate, StreamInstant, StreamConfig};
use rodio::{source::Repeat, Decoder, Source};
use rustfft::{Fft, FftDirection, num_complex::Complex32, algorithm::Radix4};
use serde::{Serialize, Serializer, de::value::UsizeDeserializer};
use futures::{stream::{Stream, StreamExt, Buffered, Map, ReadyChunks}, Future, FutureExt, future::{self, Ready}, TryFutureExt};
use tokio::{io::{BufReader}, sync::{oneshot::{self, Receiver}, watch, Semaphore, Notify}, time};

use async_std::{io::{Write}, fs::File, task::Poll, future::ready};
// use std::fs::File;
// use std::f32::consts::PI;
use std::{ops::{DerefMut, Deref}, borrow::{Borrow, BorrowMut}, f32::consts::PI, marker::PhantomData};
use async_std::sync::{Arc, Mutex};

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



#[derive(Debug, Clone, Copy)]
pub struct AvData {
    pub instance_intensity: [f32; crate::NUM_INSTANCES as usize],
    pub callback_time: tokio::time::Instant,
    pub playback_delay: tokio::time::Duration,
}

// #[derive(Serialize)]
// pub struct ProcessedData {
//     sample_rate: u32,
//     stft_output_db: Vec<Vec<f32>>,
//     pub instance_intensity: Vec<[f32; crate::NUM_INSTANCES as usize]>,
// }

fn start_resample_loop(to_data_cb: watch::Sender<Ready<f32>>, mut from_decoder: watch::Receiver<Vec<Ready<f32>>>, from_rate: u32, to_rate: u32, channels: u32) {

    tokio::spawn( async move {
        loop {
            match from_decoder.changed().await {
                Ok(_) => {
                    let result = resample(
                        from_rate, 
                        to_rate, 
                        channels, 
                        from_decoder.borrow().as_slice(),
                    ).expect("could not resample");

                    for elt in result.iter() {
                        to_data_cb.send(*elt).expect("couldn't send");
                    }
                },
                Err(_) => panic!(),
            }
        }
    });
}

fn start_decode_loop(to_processor: watch::Sender<Ready<f32>>, to_resampler: watch::Sender<Vec<Ready<f32>>>, buffer_capacity: usize, decoder: Decoder<std::fs::File>) {
    tokio::spawn(async move {
        let mut decoder = AsyncDecoder { inner: decoder, _phantom: PhantomData }.chunks(buffer_capacity);
        loop {
            match decoder.next().await {
                Some(data) => {
                    for &elt in data.iter() {
                        to_processor.send(elt.await).expect("couldn't send");
                    }
                    to_resampler.send(data).expect("couldn't send");
                },

                // TODO maybe don't keep reading next, but go to sleep
                None => (),
            }
        }
    });
}

#[derive(Debug)]
struct ProcessedData {
    sample_offset: usize,
    instance_intensity: [f32; crate::NUM_INSTANCES as usize],
}

fn start_process_loop(to_output: watch::Sender<ProcessedData>, mut from_decoder: watch::Receiver<Ready<f32>>, sample_rate: u32, channels: u32) {
    tokio::spawn( async move {
        // let sem = Semaphore::new(2);
        let mut fft_buffer = [Complex32::from(0.0f32); FFT_SIZE];
        let mut fft_scratch = [Complex32::from(0.0f32); FFT_SIZE];
        let fft_handle = Radix4::new(FFT_SIZE, FftDirection::Forward);
        let a0 = 25.0 / 46.0;
        // let from_decoder_stream = from_decoder.into_stream();

        // let start_time = std::time::Instant::now();
        let mut sample_offset = 0;
        loop {
            for i in 0..WINDOW_SIZE {
                // wait until the other loop reads the next sample first
                // notify.notified().await;
                // from_decoder.changed().await.expect("couldn't receive from decoder");

                let sample = match channels {
                    1 => {
                        from_decoder.changed().await.expect("couldn't receive");
                        from_decoder.borrow().deref().clone()
                    },
                    2 => {
                        // get the first half of the sample
                        let sample1 = {
                            from_decoder.changed().await.expect("couldn't receive");
                            from_decoder.borrow().deref().clone()
                        };

                        let sample2 = {
                            from_decoder.changed().await.expect("couldn't receive");
                            from_decoder.borrow().deref().clone()
                        };

                        (sample1 + sample2) / 2.0

                    },
                    _ => panic!(),
                };

                let windowed_val = sample * (a0 - (1.0 - a0) * (2.0 * PI * i as f32 / WINDOW_SIZE as f32).cos());
                fft_buffer[i] = Complex32::from(windowed_val);
            }

            for i in WINDOW_SIZE..FFT_SIZE {
                fft_buffer[i] = Complex32::from(0.0);
            }

            fft_handle.process_with_scratch(&mut fft_buffer, &mut fft_scratch);

            let mut fft_mag_db = [0.0f32; FFT_SIZE/2];
            for i in 0..(FFT_SIZE/2) {
                fft_mag_db[i] = 20.0 * fft_buffer[i].norm().log10();
                if fft_mag_db[i] > MAX_DB {
                    fft_mag_db[i] = MAX_DB;
                }
            }

            sample_offset += HOP_SIZE;

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

            to_output.send(ProcessedData {
                instance_intensity,
                sample_offset,
            }).expect("couldn't send");
        }
    });

    // let out_path = std::env::current_dir().unwrap()
    //     .as_path().join("res")
    //     .join("sounds")
    //     .join(std::path::Path::new(filename).with_extension("pickle"));

    // let mut out_fd = std::fs::File::create(out_path).unwrap();

    // let pickled_data = serde_pickle::to_vec(&processed_data, Default::default()).unwrap();
    // out_fd.write(pickled_data.as_slice()).unwrap();
    // out_fd.flush().unwrap();

    // self.processed_data = processed_data;

    // println!("processing took {} ms ", start_time.elapsed().as_millis());
}

struct AsyncDecoder<'a> {
    inner: Decoder<std::fs::File>,
    _phantom: PhantomData<&'a u8>
}

impl<'a> futures::stream::Stream for AsyncDecoder<'a> {
    type Item = future::Ready<f32>;

    fn poll_next(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {

        match self.borrow_mut().inner.next() {
            Some(val) => Poll::Ready(Some(future::ready(val as f32 / i16::MAX as f32))),
            None => Poll::Pending
        }
        
    }
}

pub struct Av {
    // _device: Device,
    // _audio_stream: cpal::Stream,
    to_graphics: Arc<Mutex<watch::Sender<AvData>>>,
    from_resampler: Arc<Mutex<watch::Receiver<f32>>>,
    from_processor: Arc<Mutex<watch::Receiver<ProcessedData>>>,
}

impl Av {
    pub fn play(source_file: String) -> watch::Receiver<AvData> {
        let host = cpal::default_host();
        let device = host.default_output_device().expect("no output device available");
        let mut supported_configs_range = device.supported_output_configs()
            .expect("error while querying configs");

        let config = supported_configs_range.next()
            .expect("no supported config?!")
            .with_max_sample_rate()
            .config();
        let decoder = Decoder::new(std::fs::File::open(source_file).unwrap()).unwrap();

        let source_sample_rate = decoder.sample_rate();
        let source_channels = decoder.channels() as u32;
        let target_sample_rate = config.sample_rate.0 as u32;

        let (to_resampler_from_decoder, from_decoder_to_resampler) =  watch::channel(Vec::new());
        let (to_processor_from_decoder, from_decoder_to_processor) =  watch::channel(0.0);

        let (to_data_cb_from_resampler, from_resampler_to_data_cb) =  watch::channel(0.0);
        let (to_data_cb_from_processor, from_processor_to_data_cb) =  watch::channel(ProcessedData {
            instance_intensity: [0.0; crate::NUM_INSTANCES as usize],
            sample_offset: 0,
        });
        let (to_graphics_from_data_cb, from_data_cb_to_graphics) =  watch::channel(
            AvData {
                callback_time: time::Instant::now(),
                instance_intensity: [0.0; crate::NUM_INSTANCES as usize],
                playback_delay: time::Duration::from_secs(0),
            }
        );

        let from_resampler_to_data_cb_arc = Arc::new(Mutex::new(from_resampler_to_data_cb));
        let from_processor_to_data_cb_arc = Arc::new(Mutex::new(from_processor_to_data_cb));
        let to_graphics_from_data_cb_arc = Arc::new(Mutex::new(to_graphics_from_data_cb));

        start_decode_loop(to_processor_from_decoder, to_resampler_from_decoder, target_sample_rate as usize, decoder);
        start_resample_loop(to_data_cb_from_resampler, from_decoder_to_resampler, source_sample_rate, target_sample_rate, source_channels);
        start_process_loop(to_data_cb_from_processor, from_decoder_to_processor, source_sample_rate, source_channels);

        let output_update_sample_count = (config.sample_rate.0 / (crate::MAX_INSTANCE_UPDATE_RATE_HZ)) as usize;
        let mut current_sample_index = 0;


        let data_callback = move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
            let state = Self {
                // _device: device,
                // _audio_stream: stream,
                to_graphics: Arc::clone(&to_graphics_from_data_cb_arc),
                from_processor: Arc::clone(&from_processor_to_data_cb_arc),
                from_resampler: Arc::clone(&from_resampler_to_data_cb_arc),
            }; 
            pollster::block_on(async move {

                for i in 0..data.len() {
                    if current_sample_index % output_update_sample_count == 0 {
                        let mut receiver = state.from_processor.lock().await;
                        let av_data = match receiver.changed().await {
                            Ok(_) => {
                                AvData {
                                    instance_intensity: receiver.deref().borrow().instance_intensity,
                                    callback_time: tokio::time::Instant::from_std(std::time::Instant::now()),
                                    playback_delay: info.timestamp().playback.duration_since(&info.timestamp().callback).unwrap()
                                }
                            },
                            Err(_) => panic!(),
                        };
                        let mut sender = state.to_graphics.lock().await;
                        sender.deref_mut().send(av_data).expect("could not send");
                    }
                    
                    let mut lock = state.from_resampler.lock().await;
                    let receiver = lock.deref_mut();
                    data[i] = match receiver.changed().await {
                        Ok(_) => *receiver.deref().borrow().deref(),
                        Err(_) => panic!()
                    };

                    current_sample_index += 1;
                }
            })
        };
        let error_callback = |err| {
            panic!("{:?}", err);
        };
        let stream = device.build_output_stream(
            &config,
            data_callback,
            error_callback,
        ).unwrap();

        stream.play().unwrap();

        from_data_cb_to_graphics
    }
}

