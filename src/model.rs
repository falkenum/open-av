use anyhow::*;
use collada::PrimitiveElement;
use collada::document::{LambertEffect, MaterialEffect, LambertDiffuse};
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::{Range, Index};
use std::path::Path;
use tobj::LoadOptions;
use wgpu::util::DeviceExt;
use std::fmt::Display;

use crate::texture;

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
}

impl Vertex for ModelVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[derive(Debug)]
struct OpenFileError(&'static str);

impl Display for OpenFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for OpenFileError {}

pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: usize,
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

impl Model {
    pub fn load<P: AsRef<Path>>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path: P,
    ) -> Result<Self> {
        let res = std::env::current_dir().unwrap().as_path().join("res");
        let textures = res.join("textures");
        let p = res.join("cube.dae");
        let document = match collada::document::ColladaDocument::from_path(p.as_ref()) {
            std::result::Result::Ok(doc) => anyhow::Result::Ok(doc),
            std::result::Result::Err(s) => anyhow::Result::Err(OpenFileError(s)),
        }.unwrap();
        let mte = document.get_material_to_effect();
        let effects = document.get_effect_library();
        let objs = document.get_obj_set().unwrap();
        let images = document.get_images();

        let containing_folder = textures;
        let mut materials = Vec::new();
        let mut meshes = Vec::new();

        for obj in objs.objects {
            let diffuse_re = regex::Regex::new("(.*)-sampler").unwrap();
            let mut vertices: Vec<ModelVertex> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            let mut base_index : u32 = 0;
            let mut vertex_idx_to_data: HashMap<usize, ModelVertex> = {
                let mut result = HashMap::new();

                for i in 0..obj.vertices.len() {
                    let v = obj.vertices[i];
                    result.insert(i, ModelVertex {
                        position: [v.x as f32, v.y as f32, v.z as f32],
                        tex_coords: [0.0; 2],
                        normal: [0.0; 3],
                    });

                    vertices.push(result.get(&i).unwrap().clone());
                }
                result
            };

            for i in 0..obj.geometry.len() {
                // let mut tex_vertex_indices: Vec<(usize, usize, usize)> = Vec::new();
                // let mut normal_indices: Vec<(usize, usize, usize)> = Vec::new();
                for j in 0..obj.geometry[i].mesh.len() {
                    let elt = &obj.geometry[i].mesh[j];
                    
                    match elt {
                        PrimitiveElement::Triangles(triangles) => {
                            let material_name = triangles.material.as_ref().unwrap();
                            let effect = effects.get(mte.get(material_name).unwrap()).unwrap();

                            match effect {
                                MaterialEffect::Lambert(lambert_effect) => {
                                    // TODO don't push same material more than once
                                    match &lambert_effect.diffuse {
                                        LambertDiffuse::Texture(diffuse_sampler_name) => {
                                            let diffuse_name = diffuse_re.captures(diffuse_sampler_name).unwrap().get(1).unwrap().as_str();
                                            let diffuse_filename = images.get(diffuse_name).unwrap();
                                            let diffuse_texture =
                                                texture::Texture::load(device, queue, containing_folder.join(diffuse_filename))?;
                                            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                                layout,
                                                entries: &[
                                                    wgpu::BindGroupEntry {
                                                        binding: 0,
                                                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                                                    },
                                                    wgpu::BindGroupEntry {
                                                        binding: 1,
                                                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                                                    },
                                                ],
                                                label: None,
                                            });
                                            materials.push(Material {
                                                name: format!("{}: {} ({})", material_name, diffuse_name, diffuse_filename),
                                                diffuse_texture,
                                                bind_group,
                                            })
                                        }
                                        _ => panic!(),
                                    };
                                }
                                _ => panic!(),
                            }
                            assert_eq!(triangles.vertices.len(), triangles.tex_vertices.as_ref().unwrap().len());
                            assert_eq!(triangles.vertices.len(), triangles.normals.as_ref().unwrap().len());

                            for i in 0..triangles.vertices.len() {
                                let vertex_indices = triangles.vertices[i];
                                let tex_vertex_indices = triangles.tex_vertices.as_ref().unwrap().get(i).unwrap();
                                let normal_indices = triangles.normals.as_ref().unwrap().get(i).unwrap();
                                let mut add_vertex = |vertex_index: usize, tex_vertex_index: usize, normal_index: usize| {
                                    let tv = obj.tex_vertices[tex_vertex_index];
                                    let n = obj.normals[normal_index];
                                    indices.push(base_index + vertex_index as u32);
                                    let ModelVertex {position, tex_coords: _, normal: _} = vertex_idx_to_data.get(&vertex_index).unwrap().to_owned();
                                    let v = ModelVertex {
                                        position,
                                        tex_coords: [tv.x as f32, tv.y as f32],
                                        normal: [n.x as f32, n.y as f32, n.z as f32],
                                    };
                                    vertex_idx_to_data.insert(vertex_index, v);
                                    vertices[vertex_index] = v;
                                };

                                add_vertex(vertex_indices.0, tex_vertex_indices.0, normal_indices.0);
                                add_vertex(vertex_indices.1, tex_vertex_indices.1, normal_indices.1);
                                add_vertex(vertex_indices.2, tex_vertex_indices.2, normal_indices.2);

                            }
                            base_index += triangles.vertices.len() as u32 * 3;


                        }
                        _ => panic!()
                    }
                }
                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Vertex Buffer", path.as_ref())),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Vertex Index Buffer", path.as_ref())),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                meshes.push(Mesh {
                    vertex_buffer,
                    index_buffer,
                    num_elements: indices.len() as u32,
                    name: format!("Obj name: {}, Geometry idx: {}", obj.name, i),

                    // TODO more than one material per mesh
                    material: 0,
                });


            }
        }

        Ok(Self { meshes, materials })
    }
}

// model.rs
pub trait DrawModel<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );

    fn draw_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_mesh_instanced(mesh, material, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_bind_group(0, &material.bind_group, &[]);
        self.set_bind_group(1, camera_bind_group, &[]);
        self.set_bind_group(2, light_bind_group, &[]);

        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_model(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_model_instanced(model, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone(), camera_bind_group, light_bind_group);
        }
    }
}

// model.rs
pub trait DrawLight<'a> {
    fn draw_light_mesh(
        &mut self,
        mesh: &'a Mesh,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );

    fn draw_light_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_light_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawLight<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_light_mesh(
        &mut self,
        mesh: &'b Mesh,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_light_mesh_instanced(mesh, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, light_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_light_model(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_light_model_instanced(model, 0..1, camera_bind_group, light_bind_group);
    }
    fn draw_light_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            self.draw_light_mesh_instanced(mesh, instances.clone(), camera_bind_group, light_bind_group);
        }
    }
}
