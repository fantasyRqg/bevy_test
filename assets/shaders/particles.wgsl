// GPU Particle System - Index Buffer Optimization
//
// Compute: update particle positions
// Vertex: read particle data, generate quad vertices
// Optimization: 4 vertices + index buffer instead of 6 vertices

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    rot: f32,
    rot_vel: f32,
}

struct Params {
    dt: f32,
    bounds_x: f32,
    bounds_y: f32,
    count: u32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

// ============ COMPUTE SHADER ============
@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.count) {
        return;
    }

    var p = particles[index];

    // Update position and rotation
    p.pos += p.vel * params.dt;
    p.rot += p.rot_vel * params.dt;

    // Bounce off walls (branchless)
    let bounds = vec2<f32>(params.bounds_x, params.bounds_y);
    let clamped = clamp(p.pos, -bounds, bounds);
    let hit = vec2<f32>(
        select(1.0, -1.0, p.pos.x != clamped.x),
        select(1.0, -1.0, p.pos.y != clamped.y)
    );
    p.pos = clamped;
    p.vel *= hit;

    particles[index] = p;
}

// ============ VERTEX SHADER ============
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

const SIZE: f32 = 4.0;
const INV_HALF_SCREEN: vec2<f32> = vec2<f32>(1.0 / 640.0, 1.0 / 360.0);

// 4 corners for indexed quad (indices: 0,1,2,2,1,3)
const CORNERS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-SIZE, -SIZE),  // 0: bottom-left
    vec2<f32>( SIZE, -SIZE),  // 1: bottom-right
    vec2<f32>(-SIZE,  SIZE),  // 2: top-left
    vec2<f32>( SIZE,  SIZE),  // 3: top-right
);

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    let particle = particles[instance_index];

    // vertex_index is 0-3 for each instance (from index buffer)
    let corner = CORNERS[vertex_index];

    // Apply rotation
    let c = cos(particle.rot);
    let s = sin(particle.rot);
    let rotated_corner = vec2<f32>(
        corner.x * c - corner.y * s,
        corner.x * s + corner.y * c
    );

    let world_pos = particle.pos + rotated_corner;
    let clip_pos = world_pos * INV_HALF_SCREEN;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(clip_pos, 0.0, 1.0);

    // Color based on velocity
    let speed_sq = dot(particle.vel, particle.vel);
    let t = speed_sq * 0.0000125 + f32(instance_index) * 0.0001;
    out.color = vec4<f32>(
        0.5 + 0.5 * sin(t),
        0.3 + 0.3 * cos(t * 0.7),
        0.8,
        1.0
    );

    return out;
}

// ============ FRAGMENT SHADER ============
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
