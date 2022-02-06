
use babycat::resample::babycat_lanczos::resample;
use cpal::{Data, Sample, SampleFormat, traits::{StreamTrait, DeviceTrait, HostTrait}, Device, Host, SampleRate, StreamInstant, StreamConfig};
use log::{debug, error};
use rodio::{source::Repeat, Decoder, Source};
use rustfft::{Fft, FftDirection, num_complex::Complex32, algorithm::Radix4};
use serde::{Serialize, Serializer, de::value::UsizeDeserializer};
use futures::{stream::{Stream, StreamExt, Buffered, Map, ReadyChunks}, Future, FutureExt, future::{self, Ready}, TryFutureExt};
use tokio::{io::{BufReader}, sync::{oneshot::{self, Receiver}, watch, Semaphore, Notify}, time};

use async_std::{io::{Write}, fs::File, task::Poll, future::ready};
// use std::fs::File;
// use std::f32::consts::PI;
use std::{ops::{DerefMut, Deref}, collections::VecDeque, borrow::{Borrow, BorrowMut}, f32::consts::PI, marker::PhantomData};
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

fn start_resample_loop(to_processor: watch::Sender<Vec<f32>>, mut from_decoder: watch::Receiver<Vec<f32>>, from_rate: u32, to_rate: u32, channels: u32) {

    tokio::spawn( async move {
        loop {
            if from_decoder.has_changed().unwrap() {
                let data = from_decoder.borrow_and_update().clone();
                let result = resample(
                    from_rate, 
                    to_rate, 
                    channels, 
                    data.as_slice(),
                ).expect("could not resample");

                to_processor.send(result).expect("couldn't send");
            }
        }
    });
}

fn start_decode_loop(to_resampler: watch::Sender<Vec<f32>>, decoder: Decoder<std::fs::File>) {
    tokio::spawn(async move {
        // let channels = decoder.channels() as usize;
        let sample_rate = decoder.sample_rate() as usize;
        let mut decoder = IteratorStream::new(decoder)
            .map(|val| val as f32 / i16::MAX as f32 )
            .chunks(sample_rate); // 1 second chunks
        loop {
            match decoder.next().await {
                Some(data) => {
                    to_resampler.send(data.clone()).expect("couldn't send");
                    // error!("sending vec of size {} from decoder", data.len());
                    // to_resampler.send(data).expect("couldn't send");
                    // let now = tokio::time::Instant::now();
                    // while now.elapsed() < tokio::time::Duration::from_millis(500) {}
                },

                None => (),
            }
        }
    });
}

#[derive(Debug, Clone)]
pub struct ProcessedData {
    instance_intensity: [f32; crate::NUM_INSTANCES as usize],

    // the samples used to generate this instance_intensity
    samples: Vec<f32>,
}

fn start_process_loop(to_data_cb: watch::Sender<Vec<ProcessedData>>, mut from_resampler: watch::Receiver<Vec<f32>>, output_sample_rate: u32, channels: u32) {
    tokio::spawn( async move {
        // let sem = Semaphore::new(2);
        let mut fft_buffer = [Complex32::from(0.0f32); FFT_SIZE];
        let mut fft_scratch = [Complex32::from(0.0f32); FFT_SIZE];
        let fft_handle = Radix4::new(FFT_SIZE, FftDirection::Forward);
        let a0 = 25.0 / 46.0;
        // let mut from_decoder_stream = from_decoder; //.flat_map(|v| IteratorStream::new(v));
        // let mut input_sample_buffer = Vec::new();
        // let from_decoder_stream = from_decoder.into_stream();

        // let start_time = std::time::Instant::now();
        // let mut sample_offset = 0;
        let mut sample_queue = VecDeque::new();
        let mut processed_data_queue = VecDeque::new();
        loop {
            // from_decoder.changed().await.expect("couldn't receive");
            if from_resampler.has_changed().unwrap() {
                let samples = from_resampler.borrow_and_update();
                // assert_eq!(HOP_SIZE * channels as usize, samples.len());
                sample_queue.extend(samples.iter().cloned());
                // error!("received vec of size {} in processor", samples.len());
            }
            if sample_queue.len() >= channels as usize * WINDOW_SIZE {
                for fft_buffer_idx in 0..WINDOW_SIZE {
                    let sample = match channels {
                        1 => {
                            sample_queue.get(fft_buffer_idx as usize).unwrap().clone()
                        },
                        2 => {
                            (sample_queue.get(fft_buffer_idx * 2).unwrap() + sample_queue.get(fft_buffer_idx * 2 + 1).unwrap()) / 2.0
                        },
                        _ => panic!(),
                    };

                    let windowed_val = sample * (a0 - (1.0 - a0) * (2.0 * PI * fft_buffer_idx as f32 / WINDOW_SIZE as f32).cos());
                    fft_buffer[fft_buffer_idx] = Complex32::from(windowed_val);
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


                let max_bin = FFT_SIZE * MAX_FREQ / output_sample_rate as usize;
                // let bins_per_instance = max_bin as f32 / crate::NUM_INSTANCES as f32;

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

                let mut output_samples = Vec::new();
                for _ in 0..(HOP_SIZE * channels as usize) {
                    output_samples.push(sample_queue.pop_front().unwrap());
                }

                processed_data_queue.push_back(ProcessedData {
                    instance_intensity,
                    samples: output_samples,
                });

                // sample_offset += HOP_SIZE * channels as usize;
            }

            // if have 1 second of samples, then send
            if processed_data_queue.len() * HOP_SIZE * channels as usize > output_sample_rate as usize {
                // flush the queue
                to_data_cb.send(Vec::from_iter(processed_data_queue.iter().cloned())).expect("couldn't send");
                processed_data_queue = VecDeque::new();
            }
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

struct IteratorStream<V, I: IntoIterator<Item = V>> {
    // inner: I,
    current: <I as IntoIterator>::IntoIter
}

impl<V, I: IntoIterator<Item = V>> IteratorStream<V, I> {
    fn new(inner: I) -> Self {
        Self {
            current: inner.into_iter(),
        }
    }
}

impl<V, I: IntoIterator<Item = V>> Unpin for IteratorStream<V, I> {
}

impl<V, I: IntoIterator<Item = V>> futures::stream::Stream for IteratorStream<V, I> {
    type Item = V;

    fn poll_next(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {

        match self.current.next() {
            Some(val) => {
                // cx.waker().wake_by_ref();
                Poll::Ready(Some(val))
            }
            None => Poll::Pending
        }
        
    }
}

struct ReceiverStream<V: Clone> {
    inner: watch::Receiver<V>,
}
impl<V: Clone> Unpin for ReceiverStream<V> {}

impl<V: Clone> futures::stream::Stream for ReceiverStream<V> {
    type Item = V;

    fn poll_next(mut self: std::pin::Pin<&mut Self>, _: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {

        if self.inner.has_changed().unwrap() {
            Poll::Ready(Some(self.inner.borrow_and_update().clone()))
        } else {
            Poll::Pending
        }
    }
}

pub struct AvState {
    // _device: Device,
    // processed_data_queue: VecDeque<ProcessedData>,
    processed_data_sample_idx: usize,
    processed_data_queue: VecDeque<ProcessedData>,
    to_graphics: watch::Sender<AvData>,
    from_processor: watch::Receiver<Vec<ProcessedData>>,
    // from_processor: watch::Receiver<ProcessedData>,
}

pub struct Av {
    _state: Arc<Mutex<AvState>>,
    _stream: cpal::Stream,
    pub av_data_receiver: Arc<Mutex<watch::Receiver<AvData>>>,
}

impl Av {
    pub async fn play(source_file: String) -> Self {
        // let (jack_client, jack_client_status) = jack::Client::new("open-av jack client", jack::ClientOptions::empty()).unwrap();
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
        let default_processed_data = ProcessedData {
            instance_intensity: [0.0; crate::NUM_INSTANCES as usize],
            samples: Vec::new(),
        };

        let (to_resampler_from_decoder, from_decoder_to_resampler) =  watch::channel(Vec::new());
        let (to_processor_from_resampler, from_resampler_to_processor) =  watch::channel(Vec::new());
        let (to_data_cb_from_processor, from_processor_to_data_cb) =  watch::channel(Vec::new());

        let (to_graphics_from_data_cb, from_data_cb_to_graphics) =  watch::channel(
            AvData {
                callback_time: time::Instant::now(),
                instance_intensity: [0.0; crate::NUM_INSTANCES as usize],
                playback_delay: time::Duration::from_secs(0),
            }
        );

        start_decode_loop(to_resampler_from_decoder, decoder);
        start_resample_loop(to_processor_from_resampler, from_decoder_to_resampler, source_sample_rate, target_sample_rate, source_channels);
        start_process_loop(to_data_cb_from_processor, from_resampler_to_processor, target_sample_rate, source_channels);

        // let output_update_sample_count = (config.sample_rate.0 / (crate::MAX_INSTANCE_UPDATE_RATE_HZ)) as usize;
        // let mut current_sample_index: usize = 0;

        let state = Arc::new(Mutex::new(AvState {
            processed_data_sample_idx: 0,
            processed_data_queue: VecDeque::new(),
            to_graphics: to_graphics_from_data_cb,
            from_processor: from_processor_to_data_cb,
        })); 
        let state_clone = Arc::clone(&state);
        // let state_clone = Arc::clone(&state);

        let data_callback = move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
            let state= Arc::clone(&state);
            let lock = state.try_lock();
            if let None = lock {
                return;
            };
            let mut lock = lock.unwrap();
            let state = lock.deref_mut();
            
            let receiver = &mut state.from_processor;
            if receiver.has_changed().unwrap() {
                state.processed_data_queue.extend(receiver.borrow_and_update().iter().cloned());
            }

            // if we have enough data to fill the requested buffer (this should be true almost always)
            if state.processed_data_queue.len() * HOP_SIZE * source_channels as usize >= data.len() {
                for i in 0..data.len() {
                    data[i] = state.processed_data_queue.front().unwrap().samples[state.processed_data_sample_idx];
                    state.processed_data_sample_idx += 1;
                    if state.processed_data_sample_idx >= state.processed_data_queue.front().unwrap().samples.len() {
                        let av_data = AvData {
                            instance_intensity: state.processed_data_queue.front().unwrap().instance_intensity,
                            callback_time: tokio::time::Instant::now(),
                            playback_delay: info.timestamp().playback.duration_since(&info.timestamp().callback).unwrap(),
                        };
                    
                        state.to_graphics.send(av_data).expect("couldn't send");
                        state.processed_data_queue.pop_front().unwrap();
                        state.processed_data_sample_idx = 0;
                    }
                }
            }
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

        Self {
            _state: state_clone,
            _stream: stream,
            av_data_receiver: Arc::new(Mutex::new(from_data_cb_to_graphics)),
        }
    }
}


