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

mod mesh;
mod texture;
mod camera;
mod av;

use mesh::{DrawMesh, Vertex};
// const NUM_INSTANCES_PER_ROW: u32 = 4;
const NUM_INSTANCES: u32 = 32;
const FPS: u32 = 60;

// main.rs
struct Context {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    last_frame_update: tokio::time::Instant,
    av: av::Av,
    av_data_queue: Arc<Mutex<VecDeque<av::AvData>>>,
    mesh: mesh::Mesh,
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

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = {
            let shader = include_wgsl!("shader.wgsl");
            let shader = device.create_shader_module(&shader);

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[mesh::MeshVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[wgpu::ColorTargetState {
                        format: config.format,
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
        };
        let source_path = std::env::current_dir()
            .unwrap().as_path().join("res").join("sounds").join("enter-sandman.wav");

        let source_file = source_path.to_str().unwrap();

        let av = av::Av::play(String::from(source_file)).await;

        Self {
            mesh: mesh::Mesh::new(&device),
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            last_frame_update: tokio::time::Instant::now(),
            av_data_queue: Arc::new(Mutex::new(VecDeque::new())),
            av,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            // self.camera_context.on_resize(self.config.width as f32, self.config.height as f32);
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            // self.depth_texture =
            //     texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        // self.camera_context.on_input(event)
        true
    }

    fn update(&mut self) {
        // self.camera_context.update();

        // pollster::block_on(async {
        //     let mut queue_lock = self.av_data_queue.lock().await;
        //         // find the last processed data that corresponds to before the playback time
        //     while let Some(av_data) = queue_lock.front() {
        //         let av_data = av_data.clone();
        //         if av_data.callback_time.elapsed() + tokio::time::Duration::from_millis(30) > av_data.playback_delay {
        //             queue_lock.pop_front().unwrap();
        //             for i in 0..self.instances.len() {
        //                 self.instances[i].pose[1][3] = 25.0 * av_data.instance_intensity[i];
        //             }
        //         } else {
        //             break;
        //         }
        //     }
        // });
        // self.queue.write_buffer(&self.mesh, 0, bytemuck::cast_slice(&self.instances.deref()));

        // let mesh_vertices = [
        //     mesh::MeshVertex {
        //         position: [0., 0., 0.],
        //         color: [1., 1., 1., 1.],
        //     },
        //     mesh::MeshVertex {
        //         position: [1., 1., 0.],
        //         color: [1., 1., 1., 1.],
        //     },
        //     mesh::MeshVertex {
        //         position: [1., -1., 0.],
        //         color: [1., 1., 1., 1.],
        //     },
        // ];

        // self.queue.write_buffer(&self.mesh.vertex_buffer, 0, bytemuck::cast_slice(&mesh_vertices));
        // self.queue.write_buffer(&self.mesh.index_buffer, 0, bytemuck::cast_slice(&[0, 1, 2]));
        // self.mesh.num_elements = 1;
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
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_mesh(&self.mesh)
            // render_pass.draw(0..3, 0..1)

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
