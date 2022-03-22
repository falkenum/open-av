use std::{ops::{DerefMut, Deref}, iter, collections::VecDeque, time::Duration, borrow::Borrow};

use async_std::{sync::{Arc, Mutex}, channel::Send};
use image::GenericImageView;
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
    camera_bind_group: wgpu::BindGroup,
    diffuse_bind_group: wgpu::BindGroup,
    // camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_uniform_queue: VecDeque<CameraUniform>,
    camera_uniform: CameraUniform,
    next_cu_idx: usize,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    origin: [f32; 3],
    scale: f32,
    layer: u32,
    _pad: [u32; 3],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            origin: [0.0, 0.0, 0.0],
            scale: 1.0,
            layer: 0,
            _pad: [0; 3],
        }
    }
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

        let camera_uniform = CameraUniform::new();
        // uniform.update_view_proj(&camera);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
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

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: None,
        });

        let mut pixel_data = vec![];

        // want 7 visible layers of fractal
        for i in 0..7 {
            let val = (i as f32 / 7.0 * 255.0) as u8;
            pixel_data.push(val);
            pixel_data.push(val);
            pixel_data.push(val);
            pixel_data.push(255);
        }

        let diffuse_image = image::DynamicImage::ImageRgba8(image::ImageBuffer::from_vec(pixel_data.len() as u32 / 4, 1, pixel_data).unwrap());
        let texture_size = wgpu::Extent3d {
            width: diffuse_image.dimensions().0,
            height: diffuse_image.dimensions().1,
            depth_or_array_layers: 1,
        };

        let diffuse_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                // All textures are stored as 3D, we represent our 2D texture
                // by setting depth to 1.
                size: texture_size,
                mip_level_count: 1, // We'll talk about this a little later
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // Most images are stored using sRGB so we need to reflect that here.
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
                // COPY_DST means that we want to copy data to this texture
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: None,
            }
        );

        queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            // The actual pixel data
            diffuse_image.as_rgba8().unwrap(),
            // The layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * diffuse_image.dimensions().0),
                rows_per_image: std::num::NonZeroU32::new(diffuse_image.dimensions().1),
            },
            texture_size,
        );

        let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let texture_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
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
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
                label: None,
            }
        );
        
        let diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                    }
                ],
                label: None,
            }
        );

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &texture_bind_group_layout],
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
        
        let camera_uniform_queue = VecDeque::new();
        let mesh = mesh::Mesh::new(&device);

        
        let camera_uniform = CameraUniform::new();

        Self {
            mesh,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            camera_bind_group,
            last_frame_update: tokio::time::Instant::now(),
            av_data_queue: Arc::new(Mutex::new(VecDeque::new())),
            av,
            camera_buffer,
            camera_uniform_queue,
            camera_uniform,
            next_cu_idx: 0,
            diffuse_bind_group,
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
        false
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

        let anim_len = 2.0;
        if self.camera_uniform_queue.len() == 0 {
            let num_steps = (anim_len * FPS as f32).ceil() as u32;
            let next_zoom_point = self.mesh.fractal.zoom_points[self.next_cu_idx];

            let origin_diff = cgmath::Vector3::from(next_zoom_point.origin) - cgmath::Vector3::from(self.camera_uniform.origin);
            let origin_step = origin_diff / (anim_len * FPS as f32);
            let scale_diff = next_zoom_point.scale - self.camera_uniform.scale;
            let scale_step = scale_diff / (anim_len * FPS as f32);
            
            for _i in 0..num_steps {
                self.camera_uniform.origin = (cgmath::Vector3::from(self.camera_uniform.origin) + origin_step).into();
                self.camera_uniform.scale += scale_step;

                self.camera_uniform_queue.push_back(self.camera_uniform);
            }

            self.next_cu_idx += 1;

            if self.next_cu_idx == self.mesh.fractal.zoom_points.len() {
                self.next_cu_idx = 3;
                self.camera_uniform = self.mesh.fractal.zoom_points[2];
            } else {
                self.camera_uniform = next_zoom_point;
            }
        }

        let next = self.camera_uniform_queue.pop_front().unwrap();
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[next]));
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
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.diffuse_bind_group, &[]);
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
