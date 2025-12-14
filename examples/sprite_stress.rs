//! GPU Particle System - Index Buffer Optimization
//!
//! Architecture:
//! - Compute shader updates particle positions
//! - Vertex shader reads particles, generates quads
//! - Index buffer: 4 vertices + 6 indices instead of 6 vertices (33% fewer)

use bevy::{
    core_pipeline::core_2d::graph::Core2d,
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
            ComputePassDescriptor, ComputePipelineDescriptor, FragmentState, IndexFormat,
            MultisampleState, PipelineCache, PrimitiveState, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages, ShaderType,
            TextureFormat, VertexState,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        view::ViewTarget,
        Extract, Render, RenderApp, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};
use rand::Rng;

const SPAWN_COUNT: u32 = 100000;
const MAX_PARTICLES: u32 = 2_000_000;
const HALF_WIDTH: f32 = 640.0;
const HALF_HEIGHT: f32 = 360.0;
const SPRITE_SIZE: f32 = 4.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "GPU Particle System".into(),
                resolution: (1280u32, 720u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(GpuParticlePlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (handle_input, update_ui))
        .run();
}

// ============ Main World Resources ============

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
struct GpuParticle {
    pos: [f32; 2],
    vel: [f32; 2],
    rot: f32,      // rotation angle in radians
    rot_vel: f32,  // angular velocity in radians/sec
}

#[derive(Resource, Default)]
struct ParticleState {
    count: u32,
    pending_spawns: Vec<GpuParticle>,
    pending_removes: u32,
}

#[derive(Component, ExtractComponent, Clone)]
struct ParticleCamera;

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
    commands.spawn((Camera2d, ParticleCamera));

    commands.spawn((
        Text::new("GPU Particles: 0 | FPS: 0\nSPACE: Spawn | BACKSPACE: Remove"),
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

    commands.init_resource::<ParticleState>();
    commands.init_resource::<SmoothedFps>();
}

fn handle_input(keyboard: Res<ButtonInput<KeyCode>>, mut state: ResMut<ParticleState>) {
    state.pending_spawns.clear();
    state.pending_removes = 0;

    if keyboard.just_pressed(KeyCode::Space) {
        let mut rng = rand::rng();
        let spawn_count = SPAWN_COUNT.min(MAX_PARTICLES - state.count);

        for _ in 0..spawn_count {
            state.pending_spawns.push(GpuParticle {
                pos: [
                    rng.random_range(-HALF_WIDTH..HALF_WIDTH),
                    rng.random_range(-HALF_HEIGHT..HALF_HEIGHT),
                ],
                vel: [
                    rng.random_range(-200.0..200.0),
                    rng.random_range(-200.0..200.0),
                ],
                rot: rng.random_range(0.0..std::f32::consts::TAU),
                rot_vel: rng.random_range(-5.0..5.0),
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
    state: Res<ParticleState>,
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
            "GPU Particles: {} | FPS: {:.0}\n\
             ─────────────────────────────\n\
             SPACE: Spawn {} | BACKSPACE: Remove\n\
             Max: {}",
            state.count, smoothed_fps.value, SPAWN_COUNT, MAX_PARTICLES
        );
    }
}

// ============ GPU Particle Plugin ============

struct GpuParticlePlugin;

impl Plugin for GpuParticlePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<ParticleCamera>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(ExtractSchedule, extract_particles)
            .add_systems(Render, prepare_particles.in_set(RenderSystems::Prepare))
            .add_render_graph_node::<ViewNodeRunner<ParticleNode>>(Core2d, ParticleNodeLabel)
            .add_render_graph_edge(
                Core2d,
                bevy::core_pipeline::core_2d::graph::Node2d::EndMainPass,
                ParticleNodeLabel,
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<ParticlePipeline>();
    }
}

// ============ Render World Resources ============

#[derive(Resource, Default)]
struct ExtractedParticles {
    pending_spawns: Vec<GpuParticle>,
    pending_removes: u32,
    total_count: u32,
}

#[derive(Resource)]
struct GpuParticleBuffers {
    particle_buffer: Buffer,
    params_buffer: Buffer,
    index_buffer: Buffer,
    bind_group: BindGroup,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
struct GpuParams {
    dt: f32,
    bounds_x: f32,
    bounds_y: f32,
    count: u32,
}

fn extract_particles(mut commands: Commands, state: Extract<Res<ParticleState>>) {
    commands.insert_resource(ExtractedParticles {
        pending_spawns: state.pending_spawns.clone(),
        pending_removes: state.pending_removes,
        total_count: state.count,
    });
}

const PARTICLE_SIZE: u64 = std::mem::size_of::<GpuParticle>() as u64;

fn prepare_particles(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline: Res<ParticlePipeline>,
    extracted: Res<ExtractedParticles>,
    buffers: Option<ResMut<GpuParticleBuffers>>,
    time: Res<Time>,
) {
    match buffers {
        None => {
            // First time: create all buffers

            // Particle buffer
            let particle_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("particle_buffer"),
                size: MAX_PARTICLES as u64 * PARTICLE_SIZE,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            if !extracted.pending_spawns.is_empty() {
                queue.write_buffer(&particle_buffer, 0, bytemuck::cast_slice(&extracted.pending_spawns));
            }

            // Params buffer
            let params = GpuParams {
                dt: time.delta_secs(),
                bounds_x: HALF_WIDTH - SPRITE_SIZE / 2.0,
                bounds_y: HALF_HEIGHT - SPRITE_SIZE / 2.0,
                count: extracted.total_count,
            };
            let params_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("params_buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

            // Index buffer (static pattern: 0,1,2,2,1,3 for each quad)
            // Using instanced rendering, so we only need 6 indices
            let indices: [u32; 6] = [0, 1, 2, 2, 1, 3];
            let index_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("index_buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: BufferUsages::INDEX,
            });

            // Bind group
            let bind_group = device.create_bind_group(
                Some("particle_bind_group"),
                &pipeline.bind_group_layout,
                &BindGroupEntries::sequential((
                    particle_buffer.as_entire_buffer_binding(),
                    params_buffer.as_entire_buffer_binding(),
                )),
            );

            commands.insert_resource(GpuParticleBuffers {
                particle_buffer,
                params_buffer,
                index_buffer,
                bind_group,
                count: extracted.total_count,
            });
        }
        Some(mut buffers) => {
            // Append new spawns
            if !extracted.pending_spawns.is_empty() {
                let write_offset = (buffers.count - extracted.pending_removes) as u64 * PARTICLE_SIZE;
                queue.write_buffer(
                    &buffers.particle_buffer,
                    write_offset,
                    bytemuck::cast_slice(&extracted.pending_spawns),
                );
            }

            buffers.count = extracted.total_count;

            // Update params
            let params = GpuParams {
                dt: time.delta_secs(),
                bounds_x: HALF_WIDTH - SPRITE_SIZE / 2.0,
                bounds_y: HALF_HEIGHT - SPRITE_SIZE / 2.0,
                count: buffers.count,
            };
            queue.write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));
        }
    }
}

// ============ Pipeline ============

#[derive(Resource)]
struct ParticlePipeline {
    bind_group_layout: BindGroupLayout,
    compute_pipeline: CachedComputePipelineId,
    render_pipeline_id: CachedRenderPipelineId,
    render_pipeline_hdr_id: CachedRenderPipelineId,
}

impl FromWorld for ParticlePipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let shader = asset_server.load("shaders/particles.wgsl");

        // Bind group layout: particles + params (shared by compute and render)
        let bind_group_layout = device.create_bind_group_layout(
            Some("particle_bind_group_layout"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE | ShaderStages::VERTEX,
                (
                    storage_buffer::<GpuParticle>(false),
                    uniform_buffer::<GpuParams>(false),
                ),
            ),
        );

        // Compute pipeline
        let compute_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("particle_compute".into()),
            layout: vec![bind_group_layout.clone()],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("update".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        // Render pipelines
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

impl ParticlePipeline {
    fn render_pipeline_descriptor(
        bind_group_layout: BindGroupLayout,
        shader: Handle<Shader>,
        format: TextureFormat,
    ) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some(format!("particle_render_{format:?}").into()),
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
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
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
struct ParticleNodeLabel;

#[derive(Default)]
struct ParticleNode;

impl ViewNode for ParticleNode {
    type ViewQuery = &'static ViewTarget;

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        view_target: QueryItem<'w, 'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(buffers) = world.get_resource::<GpuParticleBuffers>() else {
            return Ok(());
        };

        if buffers.count == 0 {
            return Ok(());
        }

        let pipeline = world.resource::<ParticlePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let encoder = render_context.command_encoder();

        // Compute pass: update particle positions
        if let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.compute_pipeline) {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("particle_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(compute_pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            pass.dispatch_workgroups((buffers.count + 63) / 64, 1, 1);
        }

        // Render pass: draw particles with index buffer
        let render_pipeline_id = pipeline.get_render_pipeline_id(view_target.is_hdr());
        if let Some(render_pipeline) = pipeline_cache.get_render_pipeline(render_pipeline_id) {
            let color_attachment = RenderPassColorAttachment {
                view: view_target.main_texture_view(),
                resolve_target: None,
                ops: bevy::render::render_resource::Operations {
                    load: bevy::render::render_resource::LoadOp::Load,
                    store: bevy::render::render_resource::StoreOp::Store,
                },
                depth_slice: None,
            };

            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("particle_render_pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(render_pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            pass.set_index_buffer(*buffers.index_buffer.slice(..), IndexFormat::Uint32);

            // Draw 6 indices per instance, buffers.count instances
            pass.draw_indexed(0..6, 0, 0..buffers.count);
        }

        Ok(())
    }
}
