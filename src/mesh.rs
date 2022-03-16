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

struct Fractal {
    // counter-clockwise triangle vertices
    base: [cgmath::Vector3<f32>; 3],
    vertices: Vec<cgmath::Vector3<f32>>,
    radius_scale: f32,
}

impl Fractal {
    fn new(num_layers: u32) -> Self {
        let q120 = cgmath::Quaternion::from_angle_z(cgmath::Deg(120.0f32));

        let r = 0.8;
        let top = cgmath::vec3(0., r, 0.);
        let left = q120.rotate_vector(top);
        let right = q120.rotate_vector(left);

        let x0 = 0.0;
        let y0 = 0.0;

        let x1 = top[0];
        let y1 = top[1];

        let x2 = right[0];
        let y2 = right[1];

        let dist_to_line = ((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1)).abs() / ((x2 - x1).powf(2.) + (y2 - y1).powf(2.)).sqrt();
        let scale = dist_to_line / r; 

        // let base_to_center_q = cgmath::Quaternion::from_angle_z(cgmath::Deg(180.0f32));
        let mut result = Self {
            base: [
                top,
                left,
                right,
            ],
            vertices: Vec::new(),
            radius_scale: scale,
        };

        result.add_black_centers(cgmath::vec3(0., 0., 0.), dist_to_line, num_layers);

        result
    }

    fn add_black_centers(&mut self, origin: cgmath::Vector3<f32>, radius: f32, depth: u32) {
        let q120 = cgmath::Quaternion::from_angle_z(cgmath::Deg(120.0f32));
        let q60 = cgmath::Quaternion::from_angle_z(cgmath::Deg(60.0f32));

        if depth == 0 {
            return 
        }

        let bottom = cgmath::vec3(0., -radius, 0.);
        let right = q120.rotate_vector(bottom);
        let left = q120.rotate_vector(right);

        self.vertices.push(bottom + origin);
        self.vertices.push(right + origin);
        self.vertices.push(left + origin);

        let child_origin = q60.rotate_vector(bottom);
        self.add_black_centers(child_origin + origin, self.radius_scale * radius, depth - 1);
        let child_origin = q60.rotate_vector(right);
        self.add_black_centers(child_origin + origin, self.radius_scale * radius, depth - 1);
        let child_origin = q60.rotate_vector(left);
        self.add_black_centers(child_origin + origin, self.radius_scale * radius, depth - 1);
    }

    fn into_mesh_vertices(self) -> Vec<MeshVertex> {
        let mut result = Vec::new();
        
        for v in self.base {
            result.push(MeshVertex {
                color: [1., 1., 1., 1.],
                position: v.into()
            });
        }

        for v in self.vertices {
            result.push(MeshVertex {
                color: [0., 0., 0., 1.],
                position: v.into(),
            });
        }

        result
    }
}

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

impl Mesh {
    pub fn new(device: &wgpu::Device) -> Self {

        let fractal = Fractal::new(8);
        let mesh_vertices = fractal.into_mesh_vertices();

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