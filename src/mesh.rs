use core::num;

use cgmath::{Rotation3, Rotation, num_traits::Float};
use wgpu::util::DeviceExt;


pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl Vertex for MeshVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },

                // color
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

impl Mesh {
    pub fn new(device: &wgpu::Device) -> Self {
        let white = [1., 1., 1., 1.];
        let black = [0., 0., 1., 1.];

        // let white_long_len = 1.0;
        // let white_short_len = (white_long_len.powf(2.0) / 2.0).sqrt();

        let q = cgmath::Quaternion::from_angle_z(cgmath::Deg(120.0f32));

        let r = 0.8;
        let tip = [0., r, 0.];
        let left = q.rotate_vector(cgmath::Vector3::from(tip)).into();
        let right = q.rotate_vector(cgmath::Vector3::from(left)).into();

        let mut mesh_vertices = vec![
            // white base triangle
            MeshVertex {
                position: tip,
                color: white.clone(),
            },
            MeshVertex {
                position: left,
                color: white.clone(),
            },
            MeshVertex {
                position: right,
                color: white.clone(),
            },
        ];

        let x0 = 0.0;
        let y0 = 0.0;

        let x1 = tip[0];
        let y1 = tip[1];

        let x2 = right[0];
        let y2 = right[1];

        let dist_to_line = ((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1)).abs() / ((x2 - x1).powf(2.) + (y2 - y1).powf(2.)).sqrt();
        let scale = dist_to_line / r; 

        let q = cgmath::Quaternion::from_angle_z(cgmath::Deg(180.0f32));

        for i in 0..mesh_vertices.len() {
            mesh_vertices.push(MeshVertex {
                position: (q.rotate_vector(cgmath::Vector3::from(mesh_vertices[i].position)) * scale).into(),
                color: black.clone(),
            });
        }

        // let num_layers = 1;
        // for i in 0..num_layers {
        //     for t in 0..(3u32.pow(i)) {
                
        //     }
        // }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&mesh_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice((0..mesh_vertices.len() as u32).collect::<Vec<u32>>().as_slice()),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        let num_indices = mesh_vertices.len() as u32;
        Mesh {
            vertex_buffer,
            index_buffer,
            num_indices,
        }
    }
}

// model.rs
pub trait DrawMesh<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
    );
}

impl<'a, 'b> DrawMesh<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.draw_indexed(0..mesh.num_indices, 0, 0..1);
    }
}