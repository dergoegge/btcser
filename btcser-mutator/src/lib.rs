pub mod mutator;
pub mod sampler;

pub use mutator::*;

use std::ffi::CStr;
use std::os::raw::{c_char, c_uchar, c_uint, c_ulonglong};

use crate::sampler::ChaoSampler;
use btcser::parser::DescriptorParser;

#[repr(C)]
pub struct CMutator {
    mutator: Mutator<'static>,
}

#[no_mangle]
pub extern "C" fn btcser_mutator_new(
    descriptor: *const c_char,
    descriptor_name: *const c_char,
) -> *mut CMutator {
    if descriptor.is_null() || descriptor_name.is_null() {
        return std::ptr::null_mut();
    }

    let (descriptor_str, name_str) = unsafe {
        match (
            CStr::from_ptr(descriptor).to_str(),
            CStr::from_ptr(descriptor_name).to_str(),
        ) {
            (Ok(d), Ok(n)) => (d, n),
            _ => return std::ptr::null_mut(),
        }
    };

    let mut parser = DescriptorParser::new();
    let descriptor = match parser.parse_file(descriptor_str) {
        Ok(_) => match parser.get_descriptor(name_str) {
            Some(d) => d.clone(),
            None => {
                eprintln!("Descriptor '{}' not found", name_str);
                return std::ptr::null_mut();
            }
        },
        Err(err) => {
            eprintln!("Descriptor didn't parse: {}", err);
            return std::ptr::null_mut();
        }
    };

    let mutator = Mutator::new(descriptor, Box::leak(Box::new(parser)));
    let cmutator = CMutator { mutator };
    Box::into_raw(Box::new(cmutator))
}

#[no_mangle]
pub extern "C" fn btcser_mutator_free(ptr: *mut CMutator) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

extern "C" {
    fn LLVMFuzzerMutate(data: *mut u8, size: usize, max_size: usize) -> usize;
}

struct LibFuzzerByteArrayMutator;

impl ByteArrayMutator for LibFuzzerByteArrayMutator {
    fn new(_seed: u64) -> Self {
        Self {}
    }

    fn mutate(&mut self, bytes: &mut Vec<u8>) {
        if bytes.capacity() == 0 {
            bytes.reserve(512);
        }
        let len = bytes.len();
        bytes.reserve(len * 2);
        let max_len = bytes.capacity();
        let mut_ptr = bytes.as_mut_ptr();
        bytes.resize(unsafe { LLVMFuzzerMutate(mut_ptr, len, max_len) }, 0);
    }

    fn mutate_in_place(&mut self, bytes: &mut [u8]) {
        let len = bytes.len();
        let mut_ptr = bytes.as_mut_ptr();
        unsafe { LLVMFuzzerMutate(mut_ptr, len, len) };
    }
}

#[no_mangle]
pub extern "C" fn btcser_mutator_mutate(
    mutator: *const CMutator,
    data: *const c_uchar,
    data_len: c_uint,
    seed: c_ulonglong,
    out: *mut *mut c_uchar,
    out_len: *mut c_uint,
) {
    if mutator.is_null() || data.is_null() || out_len.is_null() {
        unsafe {
            *out = std::ptr::null_mut();
        }
        return;
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, data_len as usize) };

    let result = unsafe {
        (*mutator)
            .mutator
            .mutate::<ChaoSampler<_>, StdSerializedValueMutator<LibFuzzerByteArrayMutator>>(
                data_slice, seed,
            )
    };

    match result {
        Ok(mutated) => {
            let mut output = Vec::with_capacity(mutated.len());
            output.extend_from_slice(&mutated);

            unsafe {
                *out_len = mutated.len() as c_uint;
                *out = output.as_mut_ptr();
            }
            std::mem::forget(output);
        }
        Err(_) => unsafe {
            *out = std::ptr::null_mut();
        },
    }
}

#[no_mangle]
pub extern "C" fn btcser_mutator_cross_over(
    mutator: *const CMutator,
    data1: *const c_uchar,
    data1_len: c_uint,
    data2: *const c_uchar,
    data2_len: c_uint,
    seed: c_ulonglong,
    out: *mut *mut c_uchar,
    out_len: *mut c_uint,
) {
    if mutator.is_null() || data1.is_null() || data2.is_null() || out.is_null() || out_len.is_null()
    {
        unsafe {
            *out = std::ptr::null_mut();
        }
        return;
    }

    let data1_slice = unsafe { std::slice::from_raw_parts(data1, data1_len as usize) };
    let data2_slice = unsafe { std::slice::from_raw_parts(data2, data2_len as usize) };

    let result = unsafe {
        (*mutator)
            .mutator
            .cross_over::<ChaoSampler<_>, ChaoSampler<_>, StdSerializedValueMutator<LibFuzzerByteArrayMutator>>(
                data1_slice,
                data2_slice,
                seed,
            )
    };

    match result {
        Ok(crossed) => {
            let mut output = Vec::with_capacity(crossed.len());
            output.extend_from_slice(&crossed);

            unsafe {
                *out_len = crossed.len() as c_uint;
                *out = output.as_mut_ptr();
            }
            std::mem::forget(output);
        }
        Err(_) => unsafe {
            *out = std::ptr::null_mut();
        },
    }
}

#[no_mangle]
pub extern "C" fn btcser_mutator_free_buffer(ptr: *mut c_uchar, len: c_uint) {
    if !ptr.is_null() {
        unsafe {
            Vec::from_raw_parts(ptr, len as usize, len as usize);
        }
    }
}
