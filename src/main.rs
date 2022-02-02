use std::{ops::{DerefMut, Deref}, iter, collections::VecDeque, time::Duration, borrow::Borrow};

use async_std::{sync::{Arc, Mutex}, channel::Send};
use log::error;
use tokio::{io::{AsyncReadExt, AsyncWriteExt}, time::Instant};
use cgmath::{InnerSpace, Rotation3, Zero};
use cpal::StreamInstant;
use rodio::Source;
use wgpu::{util::DeviceExt, include_wgsl};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

mod model;
mod texture;
mod camera;
mod av;

use model::{DrawModel, Vertex};
use camera::CameraContext;


fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(&shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}

// const NUM_INSTANCES_PER_ROW: u32 = 4;
const NUM_INSTANCES: u32 = 32;
const FPS: u32 = 60;
const MAX_INSTANCE_UPDATE_RATE_HZ: u32 = FPS*2;
const MAX_AUDIO_DRIFT_MS: u32 = 15;

// main.rs
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding: u32,
    color: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding2: u32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    origin: [[f32; 4]; 4], 
    pose: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

// impl Instance {
//     fn to_raw(&self) -> InstanceRaw {
//         let model =
//             cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation);
//         InstanceRaw {
//             model: model.into(),
//             normal: cgmath::Matrix3::from(self.rotation).into(),
//         }
//     }
// }


// struct InstanceRaw {
//     model: [[f32; 4]; 4],
//     normal: [[f32; 3]; 3],
// }

impl model::Vertex for Instance {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Instance>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 20]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 24]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 28]>() as wgpu::BufferAddress,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 32]>() as wgpu::BufferAddress,
                    shader_location: 13,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 35]>() as wgpu::BufferAddress,
                    shader_location: 14,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 38]>() as wgpu::BufferAddress,
                    shader_location: 15,
                    format: wgpu::VertexFormat::Float32x3,
                },
           ],
        }
    }
}


struct Context {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    // light_render_pipeline: wgpu::RenderPipeline,
    obj_model: model::Model,
    camera_context: camera::CameraContext,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    depth_texture: texture::Texture,
    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    last_frame_update: tokio::time::Instant,
    av: av::Av,
    av_data_queue: Arc<Mutex<VecDeque<av::AvData>>>,
}

trait VisualElement {
    fn on_resize(&mut self, _width: f32, _height: f32) {}
    fn on_input(&mut self, _event: &WindowEvent) -> bool { false }
    fn on_render(&mut self) {}
    fn update(&mut self) {}
}

impl Context {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::VULKAN);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                // Some(&std::path::Path::new("trace")), // Trace path
                None, // Trace path
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        surface.configure(&device, &config);
        let camera_context = CameraContext::new(&device, config.width as f32 / config.height as f32);

        // let animation_bind_group_layout =
        //     device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        //         entries: &[
        //             wgpu::BindGroupLayoutEntry {
        //                 binding: 0,
        //                 visibility: wgpu::ShaderStages::VERTEX,
        //                 ty: wgpu::BindingType::Buffer {
        //                     has_dynamic_offset: false,
        //                     ty: wgpu::BufferBindingType::,
        //                     min_binding_size: None,
        //                 },
        //                 count: None,
        //             },
        //         ],
        //         label: Some("animation_bind_group_layout")
        //     });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });


        const SPACE_BETWEEN: f32 = 3.0;

        let instances = (0..NUM_INSTANCES).map(|i| {
            let x = i as f32 * SPACE_BETWEEN - NUM_INSTANCES as f32 / 2.0 * SPACE_BETWEEN;
            Instance { 
                origin: [[1., 0., 0., x],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]],
                pose: [[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]],
                normal: [[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.]],
            }
        }).collect::<Vec<_>>();

        let light_uniform = LightUniform {
            position: [2.0, 2.0, 2.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0],
            _padding2: 0,
        };
         // We'll want to update our lights position, so we use COPY_DST
        let light_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Light VB"),
                contents: bytemuck::cast_slice(&[light_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let light_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: None,
        });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });



        let path = std::env::current_dir().unwrap()
            .as_path().join("res")
            .join("blender")
            .join("cube.dae");
        let obj_model = model::Model::load(
            &device,
            &queue,
            &texture_bind_group_layout,
            &path,
        )
        .unwrap();

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_context.bind_group_layout,
                    &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = {
            let shader = include_wgsl!("shader.wgsl");
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), Instance::desc()],
                shader,
            )
        };
        // let light_render_pipeline = {
        //     let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //         label: Some("Light Pipeline Layout"),
        //         bind_group_layouts: &[&camera_context.bind_group_layout, &light_bind_group_layout],
        //         push_constant_ranges: &[],
        //     });
        //     let shader = wgpu::ShaderModuleDescriptor {
        //         label: Some("Light Shader"),
        //         source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
        //     };
        //     create_render_pipeline(
        //         &device,
        //         &layout,
        //         config.format,
        //         Some(texture::Texture::DEPTH_FORMAT),
        //         &[model::ModelVertex::desc()],
        //         shader,
        //     )
        //
        // let source_file = "C:\\Users\\sjfal\\repos\\open-av\\res\\";
        let source_path = std::env::current_dir()
            .unwrap().as_path().join("res").join("sounds").join("enter-sandman.wav");
        // let frame_idx = Arc::from(Mutex::from(0usize));
        // let source =  AvSource::new(source_file, Arc::clone(&frame_idx));
        // let instances = Arc::new(Mutex::new(instances));
        // let instances_clone = Arc::clone(&instances);

        let source_file = source_path.to_str().unwrap();

        let av = av::Av::play(String::from(source_file)).await;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            obj_model,
            camera_context,
            instances,
            instance_buffer,
            depth_texture,

            light_uniform,
            light_buffer,
            light_bind_group,
            last_frame_update: tokio::time::Instant::now(),
            av_data_queue: Arc::new(Mutex::new(VecDeque::new())),
            av,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.camera_context.on_resize(self.config.width as f32, self.config.height as f32);
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_context.on_input(event)
    }

    fn update(&mut self) {
        self.camera_context.update();

        // Update the light
        // let old_position: cgmath::Vector3<_> = self.light_uniform.position.into();
        // self.light_uniform.position =
        //     (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(1.0))
        //         * old_position)
        //         .into();
        // self.light_uniform.color = {
        //     let mut new_color = self.light_uniform.color;
        //     new_color[2] += 0.05;
        //     if new_color[2] > 1.0 {
        //         new_color[2] = 0.0;
        //     }

        //     new_color
        // };

        // let anim = self.obj_model.meshes[0].animation.clone().unwrap();
        // // .poses;
        // // let times = self.obj_model.meshes[0].animation.unwrap().times;
        // // let elapsed = self.last_frame_update.elapsed();
        // let now = std::time::Instant::now();
        // let mut i = 0;
        // while now - self.animation_start > std::time::Duration::from_millis((anim.times[i] * 1000.0) as u64) {
        //     i += 1;

        //     if i >= anim.times.len() {
        //         self.animation_start = now;
        //         i = 0;
        //     }
        // }

        // let epsilon = Duration::from_millis(MAX_AUDIO_DRIFT_MS as u64);
        // let av_data = {
        //     // let mut queue = self.av.av_data_queue.lock();
        //     // let mut next_av_data = None;
            
        //     match self.av.next_av_data {
        //         // if there's nothing in the queue, we can't do anything
        //         None => {
        //             self.av.next_av_data = queue.pop();
        //             return Err(());
        //         },
        //         Some(ref data) => {
        //             if data.playback_delay > data.callback_time.elapsed() + epsilon {
        //                 // if we're greater than epsilon before the time that the front sample is going to play, then we can't do anything
        //                 return Err(());
        //             } else if data.playback_delay + epsilon < data.callback_time.elapsed() {
        //                 // if we're greater than epsilon after the time that the front sample should play, then move on to the next one
        //                 self.av.next_av_data = queue.pop();
        //                 return Err(());
        //             } else {
        //                 let result = data.clone();
        //                 self.av.next_av_data = queue.pop();
        //                 result

        //             }
        //         }
        //     }
        // };

        // self.av.av_data_receiver.changed().await.expect("await error");
            // error!("received av data in update()");
        // if now.duration_since(callback_time) > playback_delay {


        pollster::block_on(async {
            let mut queue_lock = self.av_data_queue.lock().await;
                // find the last processed data that corresponds to before the playback time
            while let Some(av_data) = queue_lock.front() {
                let av_data = av_data.clone();
                if av_data.callback_time.elapsed() > av_data.playback_delay {
                    queue_lock.pop_front().unwrap();
                    for i in 0..self.instances.len() {
                        self.instances[i].pose[1][3] = 25.0 * av_data.instance_intensity[i];
                    }
                } else {
                    break;
                }
            }
        });
        self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&self.instances.deref()));
        self.queue.write_buffer(&self.camera_context.buffer, 0, bytemuck::cast_slice(&[self.camera_context.uniform]));
        self.queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[self.light_uniform]));

        // for instance in self.instances.iter_mut() {
        //     instance.pose = anim.transforms[i].pose;
        //     instance.normal = anim.transforms[i].normal;
        // }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(
                &self.obj_model,
                0..NUM_INSTANCES as u32,
                &self.camera_context.bind_group,
                &self.light_bind_group,
            );

            // render_pass.set_pipeline(&self.light_render_pipeline);
            // render_pass.draw_light_model(
            //     &self.obj_model,
            //     &self.camera_context.bind_group,
            //     &self.light_bind_group,
            // );
        }

        // let render_start = std::time::Instant::now();
        self.queue.submit(iter::once(encoder.finish()));
        // println!("{:?}", render_start.elapsed());
        output.present();

        Ok(())
    }


}

#[tokio::main]
async fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let title = env!("CARGO_PKG_NAME");
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .build(&event_loop)
        .unwrap();

    let mut state = Context::new(&window).await;

    state.last_frame_update = tokio::time::Instant::now();
    let av_data_receiver = Arc::clone(&state.av.av_data_receiver);
    let av_data_queue = Arc::clone(&state.av_data_queue);

    tokio::spawn(async move {
        loop {
            let mut receiver_lock = av_data_receiver.lock().await;
            let mut queue_lock = av_data_queue.lock().await;
            if receiver_lock.has_changed().unwrap() {
                let data = receiver_lock.borrow_and_update();
                queue_lock.push_back(*data);
            }
            // error!("queue has {} elements", queue_lock.len());
        }
    });

    event_loop.run(move |event, _, control_flow| {

        if state.last_frame_update.elapsed() > std::time::Duration::from_secs(1) / FPS {
            state.last_frame_update = Instant::now();

            state.update();

            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            };
        }

        *control_flow = ControlFlow::Poll;


        match event {
            Event::MainEventsCleared => window.request_redraw(),
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    });
}
