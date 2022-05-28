use std::{borrow::Cow, str::FromStr};
use wgpu::{util::DeviceExt, BindGroupLayoutEntry};

fn main() {
    env_logger::init();
    // pollster::block_on(run());
    pollster::block_on(run());
}

async fn displayAdapters() {
    let instance = wgpu::Instance::new(wgpu::Backends::DX12);

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    // println!("{:?}", adapter.limits().max);

    println!("{:?}", adapter.get_info());
}

async fn init_gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::Backends::DX12);

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .unwrap();

    (device, queue)
}

async fn execute_gpu() -> Vec<u32> {
    let instance = wgpu::Instance::new(wgpu::Backends::DX12);

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .unwrap();

    let workgroup_size = device.limits().max_compute_workgroup_size_x;

    let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("test.wgsl"))),
    });

    let input = (1..320_000).into_iter().collect::<Vec<u32>>();

    let result_size = (input.len() as f32 / workgroup_size as f32).ceil() as u64 * std::mem::size_of::<u32>() as u64;
        
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: None,
        contents: bytemuck::cast_slice(&input),
        usage:  wgpu::BufferUsages::STORAGE,
    });

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: result_size as _,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Instantiates buffer with data (`numbers`).
    // Usage allowing the buffer to be:
    //   A storage buffer (can be bound within a bind group and thus available to a shader).
    //   The destination of a copy.
    //   The source of a copy.
    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Storage Buffer"),
        size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor 
        { 
          label: None, 
          entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(result_size),
                },
                count: None,
            }
          ]
    });


    let compute_pipeline_layout =
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });


    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "reduce",
    });

    // Instantiates the bind group, once again specifying the binding of buffers.

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage_buffer.as_entire_binding(),
            }],
    });

     // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch((input.len() as f32 / workgroup_size as f32).ceil() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, result_size);

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Gets the future representing when `staging_buffer` can be read from
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    // Awaits until `buffer_future` can be read from
    if let Ok(()) = buffer_future.await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result:Vec<u32> = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        result
    } else {
        panic!("failed to run compute on gpu!")
    }
}

// async fn execute_gpu() {
//     let (device, queue) = init_gpu().await;   
// }

async fn run() {
    let result = execute_gpu().await;
    println!("{:?}", result);
}