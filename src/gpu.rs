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
// AES S-box constant (256 values for GPU storage buffer)
// ─────────────────────────────────────────────────────────────────────────────

const SBOX: [u32; 256] = [
    // 0x00-0x0f
    0x63u32, 0x7cu32, 0x77u32, 0x7bu32, 0xf2u32, 0x6bu32, 0x6fu32, 0xc5u32,
    0x30u32, 0x01u32, 0x67u32, 0x2bu32, 0xfeu32, 0xd7u32, 0xabu32, 0x76u32,
    // 0x10-0x1f
    0xcau32, 0x82u32, 0xc9u32, 0x7du32, 0xfau32, 0x59u32, 0x47u32, 0xf0u32,
    0xadu32, 0xd4u32, 0xa2u32, 0xafu32, 0x9cu32, 0xa4u32, 0x72u32, 0xc0u32,
    // 0x20-0x2f
    0xb7u32, 0xfdu32, 0x93u32, 0x26u32, 0x36u32, 0x3fu32, 0xf7u32, 0xccu32,
    0x34u32, 0xa5u32, 0xe5u32, 0xf1u32, 0x71u32, 0xd8u32, 0x31u32, 0x15u32,
    // 0x30-0x3f
    0x04u32, 0xc7u32, 0x23u32, 0xc3u32, 0x18u32, 0x96u32, 0x05u32, 0x9au32,
    0x07u32, 0x12u32, 0x80u32, 0xe2u32, 0xebu32, 0x27u32, 0xb2u32, 0x75u32,
    // 0x40-0x4f
    0x09u32, 0x83u32, 0x2cu32, 0x1au32, 0x1bu32, 0x6eu32, 0x5au32, 0xa0u32,
    0x52u32, 0x3bu32, 0xd6u32, 0xb3u32, 0x29u32, 0xe3u32, 0x2fu32, 0x84u32,
    // 0x50-0x5f
    0x53u32, 0xd1u32, 0x00u32, 0xedu32, 0x20u32, 0xfcu32, 0xb1u32, 0x5bu32,
    0x6au32, 0xcbu32, 0xbeu32, 0x39u32, 0x4au32, 0x4cu32, 0x58u32, 0xcfu32,
    // 0x60-0x6f
    0xd0u32, 0xefu32, 0xaau32, 0xfbu32, 0x43u32, 0x4du32, 0x33u32, 0x85u32,
    0x45u32, 0xf9u32, 0x02u32, 0x7fu32, 0x50u32, 0x3cu32, 0x9fu32, 0xa8u32,
    // 0x70-0x7f
    0x51u32, 0xa3u32, 0x40u32, 0x8fu32, 0x92u32, 0x9du32, 0x38u32, 0xf5u32,
    0xbcu32, 0xb6u32, 0xdau32, 0x21u32, 0x10u32, 0xffu32, 0xf3u32, 0xd2u32,
    // 0x80-0x8f
    0xcdu32, 0x0cu32, 0x13u32, 0xecu32, 0x5fu32, 0x97u32, 0x44u32, 0x17u32,
    0xc4u32, 0xa7u32, 0x7eu32, 0x3du32, 0x64u32, 0x5du32, 0x19u32, 0x73u32,
    // 0x90-0x9f
    0x60u32, 0x81u32, 0x4fu32, 0xdcu32, 0x22u32, 0x2au32, 0x90u32, 0x88u32,
    0x46u32, 0xeeu32, 0xb8u32, 0x14u32, 0xdeu32, 0x5eu32, 0x0bu32, 0xdbu32,
    // 0xa0-0xaf
    0xe0u32, 0x32u32, 0x3au32, 0x0au32, 0x49u32, 0x06u32, 0x24u32, 0x5cu32,
    0xc2u32, 0xd3u32, 0xacu32, 0x62u32, 0x91u32, 0x95u32, 0xe4u32, 0x79u32,
    // 0xb0-0xbf
    0xe7u32, 0xc8u32, 0x37u32, 0x6du32, 0x8du32, 0xd5u32, 0x4eu32, 0xa9u32,
    0x6cu32, 0x56u32, 0xf4u32, 0xeau32, 0x65u32, 0x7au32, 0xaeu32, 0x08u32,
    // 0xc0-0xcf
    0xbau32, 0x78u32, 0x25u32, 0x2eu32, 0x1cu32, 0xa6u32, 0xb4u32, 0xc6u32,
    0xe8u32, 0xddu32, 0x74u32, 0x1fu32, 0x4bu32, 0xbdu32, 0x8bu32, 0x8au32,
    // 0xd0-0xdf
    0x70u32, 0x3eu32, 0xb5u32, 0x66u32, 0x48u32, 0x03u32, 0xf6u32, 0x0eu32,
    0x61u32, 0x35u32, 0x57u32, 0xb9u32, 0x86u32, 0xc1u32, 0x1du32, 0x9eu32,
    // 0xe0-0xef
    0xe1u32, 0xf8u32, 0x98u32, 0x11u32, 0x69u32, 0xd9u32, 0x8eu32, 0x94u32,
    0x9bu32, 0x1eu32, 0x87u32, 0xe9u32, 0xceu32, 0x55u32, 0x28u32, 0xdfu32,
    // 0xf0-0xff
    0x8cu32, 0xa1u32, 0x89u32, 0x0du32, 0xbfu32, 0xe6u32, 0x42u32, 0x68u32,
    0x41u32, 0x99u32, 0x2du32, 0x0fu32, 0xb0u32, 0x54u32, 0xbbu32, 0x16u32,
];

// ─────────────────────────────────────────────────────────────────────────────
// AES-256 Rcon constant (7 values for GPU storage buffer)
// ─────────────────────────────────────────────────────────────────────────────

const RCON: [u32; 7] = [
    0x01000000u32, 0x02000000u32, 0x04000000u32, 0x08000000u32,
    0x10000000u32, 0x20000000u32, 0x40000000u32,
];

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
    sbox_buffer:      wgpu::Buffer,
    rcon_buffer:      wgpu::Buffer,
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
        //   3 → sbox         (storage read)
        //   4 → rcon         (storage read)
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
                // sbox (AES S-box lookup table)
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // rcon (AES-256 Rcon expansion values)
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
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

        // Create SBOX storage buffer (256 × u32 = 1024 bytes, read-only)
        let sbox_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("sbox"),
            contents: bytemuck::cast_slice(&SBOX),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        // Create RCON storage buffer (7 × u32 = 28 bytes, read-only)
        let rcon_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("rcon"),
            contents: bytemuck::cast_slice(&RCON),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        Ok(Self { device, queue, pipeline, bgl, sbox_buffer, rcon_buffer, adapter_name, adapter_type })
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
                wgpu::BindGroupEntry { binding: 3, resource: self.sbox_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.rcon_buffer.as_entire_binding() },
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