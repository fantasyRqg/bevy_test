// GPU 3D Cube Stress Test
//
// Compute: update cube positions and rotations
// Vertex: read cube data, generate cube vertices with quaternion rotation
// Uses 8 vertices + 36 indices per cube instance

struct Cube {
    pos: vec3<f32>,
    vel: vec3<f32>,
    rot: vec4<f32>,      // Quaternion (x, y, z, w)
    rot_vel: vec3<f32>,  // Angular velocity (axis-angle per second)
    _pad: f32,
}

struct Params {
    view_proj: mat4x4<f32>,
    dt: f32,
    bounds_x: f32,
    bounds_y: f32,
    bounds_z: f32,
    count: u32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read_write> cubes: array<Cube>;
@group(0) @binding(1) var<uniform> params: Params;

// ============ QUATERNION HELPERS ============

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

// ============ COMPUTE SHADER ============

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.count) {
        return;
    }

    var c = cubes[index];

    // Update position
    c.pos += c.vel * params.dt;

    // Update rotation (quaternion integration)
    let angle = length(c.rot_vel) * params.dt;
    if (angle > 0.0001) {
        let axis = normalize(c.rot_vel);
        let half_angle = angle * 0.5;
        let delta_q = vec4<f32>(axis * sin(half_angle), cos(half_angle));
        c.rot = normalize(quat_mul(delta_q, c.rot));
    }

    // Bounce off walls (branchless 3D)
    let bounds = vec3<f32>(params.bounds_x, params.bounds_y, params.bounds_z);
    let clamped = clamp(c.pos, -bounds, bounds);
    let hit = vec3<f32>(
        select(1.0, -1.0, c.pos.x != clamped.x),
        select(1.0, -1.0, c.pos.y != clamped.y),
        select(1.0, -1.0, c.pos.z != clamped.z)
    );
    c.pos = clamped;
    c.vel *= hit;

    cubes[index] = c;
}

// ============ VERTEX SHADER ============

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

const CUBE_SIZE: f32 = 0.5;

// 8 corners of unit cube centered at origin
const CUBE_VERTICES: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
    vec3<f32>(-1.0, -1.0, -1.0),  // 0: back-bottom-left
    vec3<f32>( 1.0, -1.0, -1.0),  // 1: back-bottom-right
    vec3<f32>( 1.0,  1.0, -1.0),  // 2: back-top-right
    vec3<f32>(-1.0,  1.0, -1.0),  // 3: back-top-left
    vec3<f32>(-1.0, -1.0,  1.0),  // 4: front-bottom-left
    vec3<f32>( 1.0, -1.0,  1.0),  // 5: front-bottom-right
    vec3<f32>( 1.0,  1.0,  1.0),  // 6: front-top-right
    vec3<f32>(-1.0,  1.0,  1.0),  // 7: front-top-left
);

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    let cube = cubes[instance_index];

    // Get local vertex position (vertex_index is 0-7 for each instance from index buffer)
    let local_pos = CUBE_VERTICES[vertex_index] * CUBE_SIZE;

    // Apply quaternion rotation
    let rotated = quat_rotate(cube.rot, local_pos);

    // World position
    let world_pos = cube.pos + rotated;

    // Apply view-projection matrix
    let clip_pos = params.view_proj * vec4<f32>(world_pos, 1.0);

    var out: VertexOutput;
    out.clip_position = clip_pos;

    // Color based on velocity and instance
    let speed = length(cube.vel);
    let t = speed * 0.01 + f32(instance_index) * 0.0001;
    out.color = vec4<f32>(
        0.5 + 0.4 * sin(t),
        0.5 + 0.4 * cos(t * 1.3),
        0.5 + 0.4 * sin(t * 0.7 + 2.0),
        1.0
    );

    return out;
}

// ============ FRAGMENT SHADER ============

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
