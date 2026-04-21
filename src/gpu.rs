// =============================================================================
//  gpu.rs — wgpu compute engine
//
//  One GpuContext is created at startup and reused for every batch.
//
//  Per-batch pipeline
//  ──────────────────
//  1. Upload key_material  (N × 64 bytes = N × 16 u32, storage buffer)
//  2. Upload GpuParams     (cipher type + first 16 bytes of enc blob, uniform)
//  3. Allocate results     (N × u32, storage + copy_src)
//  4. Dispatch shader      (⌈N/64⌉ workgroups, 64 threads each)
//  5. Copy results → staging (map_read) buffer
//  6. Map staging, read Vec<bool>, unmap
//
//  The shader (crack.wgsl) tests whether decrypting the first 8 bytes of
//  the encrypted blob with a given key yields check_bytes[0..4] == [4..8].
//  That is the exact test OpenSSH itself performs when loading a key.
// =============================================================================

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

use crate::kdf::KM_U32S;
use crate::keyparser::Cipher;

// ─────────────────────────────────────────────────────────────────────────────
// Uniform pushed to the GPU per batch
// ─────────────────────────────────────────────────────────────────────────────

/// Must match the `Params` struct in crack.wgsl exactly.
/// All fields are u32 → no padding issues under std140/std430.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParams {
    pub num_candidates: u32,
    pub cipher_type:    u32,   // 0 = chacha20-poly1305, 1 = aes256-ctr
    // First 16 bytes of the encrypted blob packed as 4 × LE u32.
    // The shader XORs these with the keystream to recover the check bytes.
    pub enc_w0: u32,
    pub enc_w1: u32,
    pub enc_w2: u32,
    pub enc_w3: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU context — init once, reuse every batch
// ─────────────────────────────────────────────────────────────────────────────

pub struct GpuContext {
    pub device:       wgpu::Device,
    pub queue:        wgpu::Queue,
    pipeline:         wgpu::ComputePipeline,
    bgl:              wgpu::BindGroupLayout,
    pub adapter_name: String,
    pub adapter_type: String,   // "DiscreteGpu" / "IntegratedGpu" / "Cpu" …
}

impl GpuContext {
    /// Request the best available adapter, compile the WGSL shader.
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends:             wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags:                wgpu::InstanceFlags::default(),
            gles_minor_version:   wgpu::Gles3MinorVersion::Automatic,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            })
            .await
            .context("no GPU adapter found; wgpu has no suitable backend")?;

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let adapter_type = format!("{:?}", info.device_type);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             Some("kspr"),
                    required_features: wgpu::Features::empty(),
                    required_limits:   wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .context("failed to open GPU device")?;

        // Compile WGSL at startup — any shader syntax errors surface here
        let shader_src = include_str!("../shaders/crack.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("crack"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Bind group layout:
        //   0 → key_material (storage read)
        //   1 → params       (uniform)
        //   2 → results      (storage read_write)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("crack_bgl"),
            entries: &[
                // key_material
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // params uniform
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // results
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("crack_pl"),
                bind_group_layouts:   &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:       Some("crack_pipeline"),
                layout:      Some(&pipeline_layout),
                module:      &shader,
                entry_point: "main",
            });

        Ok(Self { device, queue, pipeline, bgl, adapter_name, adapter_type })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Per-batch dispatch
    // ─────────────────────────────────────────────────────────────────────────

    /// Test N passphrases on the GPU.
    ///
    /// `km_u32`   — flat key-material buffer, length = N × KM_U32S
    /// `cipher`   — selects which shader branch to execute
    /// `enc_blob` — encrypted private blob; only first 16 bytes are read
    ///
    /// Returns Vec<bool> of length N.  true = check-bytes matched.
    pub async fn run_batch(
        &self,
        km_u32:   &[u32],
        cipher:   &Cipher,
        enc_blob: &[u8],
    ) -> Result<Vec<bool>> {
        let n = km_u32.len() / KM_U32S;
        if n == 0 {
            return Ok(vec![]);
        }

        // Helper: read 4 bytes from enc_blob at offset as LE u32
        let le_u32 = |off: usize| -> u32 {
            if enc_blob.len() < off + 4 { return 0; }
            u32::from_le_bytes(enc_blob[off..off + 4].try_into().unwrap())
        };

        let cipher_type: u32 = match cipher {
            Cipher::Chacha20Poly1305 => 0,
            Cipher::Aes256Ctr        => 1,
            _                         => 0,
        };

        let params = GpuParams {
            num_candidates: n as u32,
            cipher_type,
            enc_w0: le_u32(0),
            enc_w1: le_u32(4),
            enc_w2: le_u32(8),
            enc_w3: le_u32(12),
        };

        // ── GPU buffers ───────────────────────────────────────────────────────
        let km_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("km"),
            contents: bytemuck::cast_slice(km_u32),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let result_size = (n * 4) as u64;

        let results_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("results"),
            size:               result_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("staging"),
            size:               result_size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Bind group ────────────────────────────────────────────────────────
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("crack_bg"),
            layout:  &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: km_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: results_buf.as_entire_binding() },
            ],
        });

        // ── Encode + dispatch ────────────────────────────────────────────────
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("enc"),
        });
        {
            // wgpu 0.19: ComputePassDescriptor has `timestamp_writes: None`
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:             Some("crack_pass"),
                timestamp_writes:  None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // workgroup_size in shader = 64; dispatch ⌈N/64⌉ groups
            let groups = ((n as u32) + 63) / 64;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&results_buf, 0, &staging_buf, 0, result_size);
        self.queue.submit(std::iter::once(enc.finish()));

        // ── Read back ─────────────────────────────────────────────────────────
        // Bridge the callback-based map_async into an async oneshot
        let (tx, rx) = futures_channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        staging_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });

        // Poll until the GPU has finished and the map is ready
        self.device.poll(wgpu::Maintain::Wait);

        rx.await
            .context("GPU map channel dropped unexpectedly")?
            .context("GPU buffer map_async failed")?;

        let view = staging_buf.slice(..).get_mapped_range();
        let results: Vec<bool> = bytemuck::cast_slice::<u8, u32>(&view)
            .iter()
            .map(|&v| v != 0)
            .collect();
        drop(view);
        staging_buf.unmap();

        Ok(results)
    }
}