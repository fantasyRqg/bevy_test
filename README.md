# Bevy Stress Tests

GPU-based stress test examples for Bevy.

## Examples

### sprite_stress
2D GPU particle system using compute shaders and instanced rendering.
- Up to 2M particles
- SPACE: spawn 100k particles
- BACKSPACE: remove 100k particles

```
cargo run --example sprite_stress --release
```

### cube_stress
3D instanced cube rendering stress test.

```
cargo run --example cube_stress --release
```
