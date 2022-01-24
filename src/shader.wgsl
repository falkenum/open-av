// Vertex shader

struct Camera {
    view_proj: mat4x4<f32>;
};
[[group(1), binding(0)]]
var<uniform> camera: Camera;

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] tex_coords: vec2<f32>;
    [[location(2)]] normal: vec3<f32>;
};
struct InstanceInput {
    [[location(5)]] pose_matrix_0: vec4<f32>;
    [[location(6)]] pose_matrix_1: vec4<f32>;
    [[location(7)]] pose_matrix_2: vec4<f32>;
    [[location(8)]] pose_matrix_3: vec4<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] tex_coords: vec2<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] world_position: vec3<f32>;
};
struct Light {
    position: vec3<f32>;
    color: vec3<f32>;
};
// [[group(3), binding(0)]]
// var<uniform> animation: ;

[[group(2), binding(0)]]
var<uniform> light: Light;


[[stage(vertex)]]
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let instance_pose_matrix = mat4x4<f32>(
        instance.pose_matrix_0,
        instance.pose_matrix_1,
        instance.pose_matrix_2,
        instance.pose_matrix_3,
    );
    let translation_vec = vec4<f32>(
        instance_pose_matrix[0][3], 
        instance_pose_matrix[1][3], 
        instance_pose_matrix[2][3],
        1.0,
    );
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    var normal: vec4<f32> = instance_pose_matrix * vec4<f32>(model.normal, 0.0);
    out.world_normal = normal.xyz;
    var world_position: vec4<f32> = translation_vec + vec4<f32>(model.position, 0.0);
    out.world_position = world_position.xyz;
    out.clip_position = camera.view_proj * world_position;
    return out;
}
// Fragment shader

[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]]
var s_diffuse: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    
    // We don't need (or want) much ambient light, so 0.1 is fine
    let ambient_strength = 0.1;
    let ambient_color = light.color * ambient_strength;

    let light_dir = normalize(light.position - in.world_position);

    let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;


    let result = (ambient_color + diffuse_color) * object_color.xyz;

    return vec4<f32>(result, object_color.a);
}
