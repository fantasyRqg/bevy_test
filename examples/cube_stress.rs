//! GPU 3D Cube Stress Test
//!
//! Architecture:
//! - Compute shader updates cube positions and rotations (quaternion)
//! - Vertex shader reads cubes, generates cube geometry
//! - Index buffer: 8 vertices + 36 indices per cube

use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    ecs::query::QueryItem,
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{storage_buffer, uniform_buffer},
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BlendState,
            Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages,
            CachedComputePipelineId, CachedRenderPipelineId, ColorTargetState, ColorWrites,
            CompareFunction, ComputePassDescriptor, ComputePipelineDescriptor,
            DepthBiasState, DepthStencilState, Extent3d, FragmentState, IndexFormat, MultisampleState,
            PipelineCache, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPassDepthStencilAttachment, RenderPipelineDescriptor, ShaderStages, ShaderType,
            StencilState, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsages, TextureViewDescriptor, VertexState, Face,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        view::ViewTarget,
        Extract, Render, RenderApp, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};
use rand::Rng;

const SPAWN_COUNT: u32 = 100_000;
const MAX_CUBES: u32 = 2_000_000;
const HALF_BOUNDS: f32 = 100.0;
const CUBE_SIZE: f32 = 0.5;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "3D Cube Stress Test".into(),
                resolution: (1280u32, 720u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(GpuCubePlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (handle_input, update_ui, camera_control))
        .run();
}

// ============ Main World Resources ============

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
struct GpuCube {
    pos: [f32; 3],
    _pad1: f32,         // Align to 16 bytes (WGSL vec3 alignment)
    vel: [f32; 3],
    _pad2: f32,         // Align to 16 bytes
    rot: [f32; 4],      // Quaternion (x, y, z, w)
    rot_vel: [f32; 3],  // Angular velocity
    _pad3: f32,         // Align to 16 bytes
}

#[derive(Resource, Default)]
struct CubeState {
    count: u32,
    pending_spawns: Vec<GpuCube>,
    pending_removes: u32,
}

#[derive(Component, ExtractComponent, Clone)]
struct CubeCamera;

#[derive(Component)]
struct StatsText;

#[derive(Resource)]
struct SmoothedFps {
    value: f64,
    update_timer: f32,
}

impl Default for SmoothedFps {
    fn default() -> Self {
        Self { value: 0.0, update_timer: 0.0 }
    }
}

fn setup(mut commands: Commands) {
    // 3D Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 50.0, 200.0).looking_at(Vec3::ZERO, Vec3::Y),
        CubeCamera,
    ));

    // UI Text
    commands.spawn((
        Text::new("3D Cubes: 0 | FPS: 0\nSPACE: Spawn | BACKSPACE: Remove"),
        TextFont { font_size: 20.0, ..default() },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        StatsText,
    ));

    commands.init_resource::<CubeState>();
    commands.init_resource::<SmoothedFps>();
}

fn handle_input(keyboard: Res<ButtonInput<KeyCode>>, mut state: ResMut<CubeState>) {
    state.pending_spawns.clear();
    state.pending_removes = 0;

    if keyboard.just_pressed(KeyCode::Space) {
        let mut rng = rand::rng();
        let spawn_count = SPAWN_COUNT.min(MAX_CUBES - state.count);

        for _ in 0..spawn_count {
            // Random axis for initial rotation
            let axis: [f32; 3] = [
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ];
            let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            let angle = rng.random_range(0.0..std::f32::consts::TAU);
            let half_angle = angle * 0.5;
            let sin_half = half_angle.sin();

            state.pending_spawns.push(GpuCube {
                pos: [
                    rng.random_range(-HALF_BOUNDS..HALF_BOUNDS),
                    rng.random_range(-HALF_BOUNDS..HALF_BOUNDS),
                    rng.random_range(-HALF_BOUNDS..HALF_BOUNDS),
                ],
                _pad1: 0.0,
                vel: [
                    rng.random_range(-50.0..50.0),
                    rng.random_range(-50.0..50.0),
                    rng.random_range(-50.0..50.0),
                ],
                _pad2: 0.0,
                rot: if len > 0.0001 {
                    [
                        axis[0] / len * sin_half,
                        axis[1] / len * sin_half,
                        axis[2] / len * sin_half,
                        half_angle.cos(),
                    ]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                },
                rot_vel: [
                    rng.random_range(-3.0..3.0),
                    rng.random_range(-3.0..3.0),
                    rng.random_range(-3.0..3.0),
                ],
                _pad3: 0.0,
            });
        }
        state.count += spawn_count;
    }

    if keyboard.just_pressed(KeyCode::Backspace) {
        let remove = SPAWN_COUNT.min(state.count);
        state.pending_removes += remove;
        state.count -= remove;
    }
}

fn update_ui(
    mut text_query: Query<&mut Text, With<StatsText>>,
    state: Res<CubeState>,
    diagnostics: Res<DiagnosticsStore>,
    mut smoothed_fps: ResMut<SmoothedFps>,
    time: Res<Time>,
) {
    if let Some(fps) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
    {
        const SMOOTHING: f64 = 0.1;
        smoothed_fps.value = smoothed_fps.value * (1.0 - SMOOTHING) + fps * SMOOTHING;
    }

    smoothed_fps.update_timer += time.delta_secs();
    if smoothed_fps.update_timer < 0.1 {
        return;
    }
    smoothed_fps.update_timer = 0.0;

    for mut text in &mut text_query {
        **text = format!(
            "3D Cubes: {} | FPS: {:.0}\n\
             ─────────────────────────────\n\
             SPACE: Spawn {} | BACKSPACE: Remove\n\
             Max: {}",
            state.count, smoothed_fps.value, SPAWN_COUNT, MAX_CUBES
        );
    }
}

fn camera_control(
    mut scroll_events: MessageReader<bevy::input::mouse::MouseWheel>,
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<CubeCamera>>,
) {
    use bevy::input::mouse::MouseScrollUnit;

    // Scroll to zoom
    let mut scroll_delta = 0.0;
    for event in scroll_events.read() {
        scroll_delta += match event.unit {
            MouseScrollUnit::Line => event.y * 20.0,
            MouseScrollUnit::Pixel => event.y * 0.5,
        };
    }

    // Arrow keys to rotate
    let rotation_speed = 1.5 * time.delta_secs();
    let mut yaw = 0.0;
    let mut pitch = 0.0;

    if keyboard.pressed(KeyCode::ArrowLeft) {
        yaw += rotation_speed;
    }
    if keyboard.pressed(KeyCode::ArrowRight) {
        yaw -= rotation_speed;
    }
    if keyboard.pressed(KeyCode::ArrowUp) {
        pitch += rotation_speed;
    }
    if keyboard.pressed(KeyCode::ArrowDown) {
        pitch -= rotation_speed;
    }

    for mut transform in &mut query {
        // Apply zoom
        if scroll_delta != 0.0 {
            let forward = transform.forward();
            transform.translation += forward * scroll_delta;
            let new_distance = transform.translation.length().clamp(10.0, 1000.0);
            transform.translation = transform.translation.normalize() * new_distance;
        }

        // Apply rotation (orbit around origin)
        if yaw != 0.0 || pitch != 0.0 {
            // Convert current position to spherical coordinates
            let pos = transform.translation;
            let r = pos.length();
            let mut theta = pos.z.atan2(pos.x); // horizontal angle
            let mut phi = (pos.y / r).acos();   // vertical angle from top

            // Apply rotation
            theta += yaw;
            phi = (phi - pitch).clamp(0.1, std::f32::consts::PI - 0.1);

            // Convert back to cartesian
            transform.translation = Vec3::new(
                r * phi.sin() * theta.cos(),
                r * phi.cos(),
                r * phi.sin() * theta.sin(),
            );

            // Look at origin
            transform.look_at(Vec3::ZERO, Vec3::Y);
        }
    }
}

// ============ GPU Cube Plugin ============

struct GpuCubePlugin;

impl Plugin for GpuCubePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<CubeCamera>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(ExtractSchedule, extract_cubes)
            .add_systems(Render, prepare_cubes.in_set(RenderSystems::Prepare))
            .add_render_graph_node::<ViewNodeRunner<CubeNode>>(Core3d, CubeNodeLabel)
            // Run after EndMainPass but before Tonemapping
            .add_render_graph_edge(Core3d, Node3d::EndMainPass, CubeNodeLabel);
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<CubePipeline>();
    }
}

// ============ Render World Resources ============

#[derive(Resource, Default)]
struct ExtractedCubes {
    pending_spawns: Vec<GpuCube>,
    pending_removes: u32,
    total_count: u32,
    view_proj: [[f32; 4]; 4],
}

#[derive(Resource)]
struct GpuCubeBuffers {
    cube_buffer: Buffer,
    params_buffer: Buffer,
    index_buffer: Buffer,
    bind_group: BindGroup,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
struct GpuParams {
    view_proj: [[f32; 4]; 4],
    dt: f32,
    bounds_x: f32,
    bounds_y: f32,
    bounds_z: f32,
    count: u32,
    _pad: [f32; 3],
}

fn extract_cubes(
    mut commands: Commands,
    state: Extract<Res<CubeState>>,
    cameras: Extract<Query<(&GlobalTransform, &Projection), With<CubeCamera>>>,
) {
    // Get view-projection matrix from camera
    let mut view_proj = [[0.0f32; 4]; 4];
    for (global_transform, projection) in &cameras {
        let view = global_transform.to_matrix().inverse();
        let proj = projection.get_clip_from_view();
        let vp = proj * view;
        view_proj = vp.to_cols_array_2d();
    }

    commands.insert_resource(ExtractedCubes {
        pending_spawns: state.pending_spawns.clone(),
        pending_removes: state.pending_removes,
        total_count: state.count,
        view_proj,
    });
}

const CUBE_SIZE_BYTES: u64 = std::mem::size_of::<GpuCube>() as u64;

fn prepare_cubes(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline: Res<CubePipeline>,
    extracted: Res<ExtractedCubes>,
    buffers: Option<ResMut<GpuCubeBuffers>>,
    time: Res<Time>,
) {
    match buffers {
        None => {
            // First time: create all buffers

            // Cube buffer
            let cube_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("cube_buffer"),
                size: MAX_CUBES as u64 * CUBE_SIZE_BYTES,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            if !extracted.pending_spawns.is_empty() {
                queue.write_buffer(&cube_buffer, 0, bytemuck::cast_slice(&extracted.pending_spawns));
            }

            // Params buffer
            let params = GpuParams {
                view_proj: extracted.view_proj,
                dt: time.delta_secs(),
                bounds_x: HALF_BOUNDS - CUBE_SIZE,
                bounds_y: HALF_BOUNDS - CUBE_SIZE,
                bounds_z: HALF_BOUNDS - CUBE_SIZE,
                count: extracted.total_count,
                _pad: [0.0; 3],
            };
            let params_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("params_buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

            // Index buffer for cube (36 indices for 12 triangles)
            let indices: [u32; 36] = [
                // Front face (z+)
                4, 5, 6, 6, 7, 4,
                // Back face (z-)
                1, 0, 3, 3, 2, 1,
                // Top face (y+)
                3, 7, 6, 6, 2, 3,
                // Bottom face (y-)
                4, 0, 1, 1, 5, 4,
                // Right face (x+)
                5, 1, 2, 2, 6, 5,
                // Left face (x-)
                0, 4, 7, 7, 3, 0,
            ];
            let index_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("cube_index_buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: BufferUsages::INDEX,
            });

            // Bind group
            let bind_group = device.create_bind_group(
                Some("cube_bind_group"),
                &pipeline.bind_group_layout,
                &BindGroupEntries::sequential((
                    cube_buffer.as_entire_buffer_binding(),
                    params_buffer.as_entire_buffer_binding(),
                )),
            );

            commands.insert_resource(GpuCubeBuffers {
                cube_buffer,
                params_buffer,
                index_buffer,
                bind_group,
                count: extracted.total_count,
            });
        }
        Some(mut buffers) => {
            // Append new spawns
            if !extracted.pending_spawns.is_empty() {
                let write_offset = (buffers.count - extracted.pending_removes) as u64 * CUBE_SIZE_BYTES;
                queue.write_buffer(
                    &buffers.cube_buffer,
                    write_offset,
                    bytemuck::cast_slice(&extracted.pending_spawns),
                );
            }

            buffers.count = extracted.total_count;

            // Update params
            let params = GpuParams {
                view_proj: extracted.view_proj,
                dt: time.delta_secs(),
                bounds_x: HALF_BOUNDS - CUBE_SIZE,
                bounds_y: HALF_BOUNDS - CUBE_SIZE,
                bounds_z: HALF_BOUNDS - CUBE_SIZE,
                count: buffers.count,
                _pad: [0.0; 3],
            };
            queue.write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));
        }
    }
}

// ============ Pipeline ============

#[derive(Resource)]
struct CubePipeline {
    bind_group_layout: BindGroupLayout,
    compute_pipeline: CachedComputePipelineId,
    render_pipeline_id: CachedRenderPipelineId,
    render_pipeline_hdr_id: CachedRenderPipelineId,
}

impl FromWorld for CubePipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let shader = asset_server.load("shaders/cubes.wgsl");

        // Bind group layout: cubes + params (shared by compute and render)
        let bind_group_layout = device.create_bind_group_layout(
            Some("cube_bind_group_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE | ShaderStages::VERTEX,
                (
                    storage_buffer::<GpuCube>(false),
                    uniform_buffer::<GpuParams>(false),
                ),
            ),
        );

        // Compute pipeline
        let compute_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cube_compute".into()),
            layout: vec![bind_group_layout.clone()],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("update".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        // Render pipelines (SDR and HDR)
        let render_pipeline_id = pipeline_cache.queue_render_pipeline(Self::render_pipeline_descriptor(
            bind_group_layout.clone(),
            shader.clone(),
            TextureFormat::Rgba8UnormSrgb,
        ));

        let render_pipeline_hdr_id = pipeline_cache.queue_render_pipeline(Self::render_pipeline_descriptor(
            bind_group_layout.clone(),
            shader,
            TextureFormat::Rgba16Float,
        ));

        Self {
            bind_group_layout,
            compute_pipeline,
            render_pipeline_id,
            render_pipeline_hdr_id,
        }
    }
}

impl CubePipeline {
    fn render_pipeline_descriptor(
        bind_group_layout: BindGroupLayout,
        shader: Handle<Shader>,
        format: TextureFormat,
    ) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some(format!("cube_render_{format:?}").into()),
            layout: vec![bind_group_layout],
            vertex: VertexState {
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some("vertex".into()),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader,
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                cull_mode: Some(Face::Back),
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater, // Bevy uses reverse-Z
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        }
    }

    fn get_render_pipeline_id(&self, hdr: bool) -> CachedRenderPipelineId {
        if hdr { self.render_pipeline_hdr_id } else { self.render_pipeline_id }
    }
}

// ============ Render Graph Node ============

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct CubeNodeLabel;

#[derive(Default)]
struct CubeNode;

impl ViewNode for CubeNode {
    type ViewQuery = &'static ViewTarget;

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        view_target: QueryItem<'w, 'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(buffers) = world.get_resource::<GpuCubeBuffers>() else {
            return Ok(());
        };

        if buffers.count == 0 {
            return Ok(());
        }

        let pipeline = world.resource::<CubePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // Check if both pipelines are ready
        let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.compute_pipeline) else {
            return Ok(());
        };
        let render_pipeline_id = pipeline.get_render_pipeline_id(view_target.is_hdr());
        let Some(render_pipeline) = pipeline_cache.get_render_pipeline(render_pipeline_id) else {
            return Ok(());
        };

        // Create depth texture matching render target size
        let target_size = view_target.main_texture().size();
        let device = world.resource::<RenderDevice>();
        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("cube_depth_texture"),
            size: Extent3d {
                width: target_size.width,
                height: target_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());

        let encoder = render_context.command_encoder();

        // Compute pass: update cube positions and rotations
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cube_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(compute_pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            pass.dispatch_workgroups((buffers.count + 63) / 64, 1, 1);
        }

        // Render pass: draw cubes with depth testing
        {
            let color_attachment = RenderPassColorAttachment {
                view: view_target.main_texture_view(),
                resolve_target: None,
                ops: bevy::render::render_resource::Operations {
                    load: bevy::render::render_resource::LoadOp::Load,
                    store: bevy::render::render_resource::StoreOp::Store,
                },
                depth_slice: None,
            };

            let depth_attachment = RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(bevy::render::render_resource::Operations {
                    load: bevy::render::render_resource::LoadOp::Clear(0.0), // 0.0 for reverse-Z (far)
                    store: bevy::render::render_resource::StoreOp::Discard,
                }),
                stencil_ops: None,
            };

            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("cube_render_pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: Some(depth_attachment),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(render_pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            pass.set_index_buffer(*buffers.index_buffer.slice(..), IndexFormat::Uint32);

            // Draw 36 indices per cube, buffers.count instances
            pass.draw_indexed(0..36, 0, 0..buffers.count);
        }

        Ok(())
    }
}
