use anyhow::*;
use collada::PrimitiveElement;
use collada::document::{LambertEffect, MaterialEffect, LambertDiffuse};
use std::ops::Range;
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
pub struct ModelVertex(f32, f32, f32); 
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelTexCoord(f32, f32); 
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelNormal(f32, f32, f32); 

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
            ],
        }
    }
}

impl Vertex for ModelTexCoord {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelTexCoord>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

impl Vertex for ModelNormal {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelNormal>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,

                    // TODO?
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
    pub tex_vertex_buffer: wgpu::Buffer,
    pub normals_buffer: wgpu::Buffer,

    pub vertex_index_buffer: wgpu::Buffer,
    pub tex_vertex_index_buffer: wgpu::Buffer,
    pub normal_index_buffer: wgpu::Buffer,

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
        let anim = document.get_animations();
        let bind_data = document.get_bind_data_set();
        let objs = document.get_obj_set().unwrap();
        let images = document.get_images();

        // let brick_filename = document.get_images().get("brick").unwrap();

        // let material_library = objs.material_library.unwrap();
        // let diffuse_filename = match effects.get(material_library.as_str()).unwrap() {
        //     MaterialEffect::Lambert(effect) => {
        //         match &effect.diffuse {
        //             LambertDiffuse::Texture(s) => s,
        //             _ => panic!(),
        //         }
        //     }
        //     _ => panic!(),
        // };

        // let (obj_models, obj_materials) = tobj::load_obj(
        //     path.as_ref(),
        //     &LoadOptions {
        //         triangulate: true,
        //         single_index: true,
        //         ..Default::default()
        //     },
        // )?;

        // let obj_materials = obj_materials?;

        // We're assuming that the texture files are stored with the obj file
        // let containing_folder = path.as_ref().parent().context("Directory has no parent")?;
        let containing_folder = textures;
        let mut materials = Vec::new();
        let mut meshes = Vec::new();

        for obj in objs.objects {

            let vertices: Vec<ModelVertex> = obj.vertices.iter().map(|v| -> ModelVertex {
                ModelVertex(v.x as f32, v.y as f32, v.z as f32)
            }).collect();

            let tex_vertices: Vec<ModelTexCoord> = obj.tex_vertices.iter().map(|v| -> ModelTexCoord {
                ModelTexCoord(v.x as f32, v.y as f32)
            }).collect();

            let normals: Vec<ModelNormal> = obj.normals.iter().map(|v| -> ModelNormal {
                ModelNormal(v.x as f32, v.y as f32, v.z as f32)
            }).collect();

            let process_indices = |indices: &Vec<(usize, usize, usize)>| -> Vec<u32> {
                let mut result: Vec<u32> = Vec::new();
                indices.iter().for_each(|triangle| {
                    result.push(triangle.0 as u32);
                    result.push(triangle.1 as u32);
                    result.push(triangle.2 as u32);
                });
                result
            };

            for i in 0..obj.geometry.len() {
                let mut vertex_indices: Vec<(usize, usize, usize)> = Vec::new();
                let mut tex_vertex_indices: Vec<(usize, usize, usize)> = Vec::new();
                let mut normal_indices: Vec<(usize, usize, usize)> = Vec::new();
                for elt in &obj.geometry[i].mesh {
                    match elt {
                        PrimitiveElement::Triangles(triangles) => {
                            let material_name = triangles.material.as_ref().unwrap();
                            let effect = effects.get(mte.get(material_name).unwrap()).unwrap();

                            match effect {
                                MaterialEffect::Lambert(lambert_effect) => {
                                    // TODO don't push same material more than once
                                    match &lambert_effect.diffuse {
                                        LambertDiffuse::Texture(diffuse_name) => {
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
                            vertex_indices.extend(triangles.vertices.iter());
                            tex_vertex_indices.extend(triangles.tex_vertices.as_ref().unwrap().iter());
                            normal_indices.extend(triangles.normals.as_ref().unwrap().iter());
                        }
                        _ => panic!()
                    }
                }
                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Vertex Buffer", path.as_ref())),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let tex_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Tex Vertex Buffer", path.as_ref())),
                    contents: bytemuck::cast_slice(&tex_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let normals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Normals Buffer", path.as_ref())),
                    contents: bytemuck::cast_slice(&normals),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let vertex_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Vertex Index Buffer", path.as_ref())),
                    contents: bytemuck::cast_slice(&process_indices(&vertex_indices)),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let tex_vertex_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Tex Vertex Index Buffer", path.as_ref())),
                    contents: bytemuck::cast_slice(&process_indices(&tex_vertex_indices)),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let normal_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Normal Index Buffer", path.as_ref())),
                    contents: bytemuck::cast_slice(&process_indices(&normal_indices)),
                    usage: wgpu::BufferUsages::INDEX,
                });

                meshes.push(Mesh {
                    vertex_buffer,
                    tex_vertex_buffer,
                    normals_buffer,
                    vertex_index_buffer,
                    tex_vertex_index_buffer,
                    normal_index_buffer,
                    num_elements: vertex_indices.len() as u32,
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
        self.set_index_buffer(mesh.vertex_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
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
        self.set_index_buffer(mesh.vertex_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
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
