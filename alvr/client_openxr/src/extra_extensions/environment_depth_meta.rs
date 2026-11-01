use openxr::{
    self as xr, raw,
    sys::{self, Handle},
};
use std::ptr;

pub struct EnvironmentDepthImageMETA {
    pub swapchain_index: u32,
    pub near_z: f32,
    pub far_z: f32,
    pub views: [(xr::Posef, xr::Fovf); 2],
}

// Real-time depth maps from the headset depth sensor (XR_META_environment_depth). Requires the
// com.oculus.permission.USE_SCENE runtime permission and an active passthrough.
pub struct EnvironmentDepthMETA {
    provider: sys::EnvironmentDepthProviderMETA,
    swapchain: sys::EnvironmentDepthSwapchainMETA,
    swapchain_images: Vec<u32>,
    resolution: (u32, u32),
    ext_fns: raw::EnvironmentDepthMETA,
}

impl EnvironmentDepthMETA {
    pub fn new(session: &xr::Session<xr::OpenGlEs>) -> xr::Result<Self> {
        let ext_fns = session
            .instance()
            .exts()
            .meta_environment_depth
            .ok_or(sys::Result::ERROR_EXTENSION_NOT_PRESENT)?;

        let mut provider = sys::EnvironmentDepthProviderMETA::NULL;
        let info = sys::EnvironmentDepthProviderCreateInfoMETA {
            ty: sys::EnvironmentDepthProviderCreateInfoMETA::TYPE,
            next: ptr::null(),
            create_flags: sys::EnvironmentDepthProviderCreateFlagsMETA::EMPTY,
        };
        unsafe {
            super::xr_res((ext_fns.create_environment_depth_provider)(
                session.as_raw(),
                &info,
                &mut provider,
            ))?
        };

        // From here on, the provider (and the swapchain) are destroyed by Drop on failure
        let mut this = Self {
            provider,
            swapchain: sys::EnvironmentDepthSwapchainMETA::NULL,
            swapchain_images: vec![],
            resolution: (0, 0),
            ext_fns,
        };

        let info = sys::EnvironmentDepthSwapchainCreateInfoMETA {
            ty: sys::EnvironmentDepthSwapchainCreateInfoMETA::TYPE,
            next: ptr::null(),
            create_flags: sys::EnvironmentDepthSwapchainCreateFlagsMETA::EMPTY,
        };
        unsafe {
            super::xr_res((ext_fns.create_environment_depth_swapchain)(
                provider,
                &info,
                &mut this.swapchain,
            ))?
        };

        let mut state = sys::EnvironmentDepthSwapchainStateMETA::out(ptr::null_mut());
        let state = unsafe {
            super::xr_res((ext_fns.get_environment_depth_swapchain_state)(
                this.swapchain,
                state.as_mut_ptr(),
            ))?;

            state.assume_init()
        };
        this.resolution = (state.width, state.height);

        let mut image_count = 0;
        unsafe {
            super::xr_res((ext_fns.enumerate_environment_depth_swapchain_images)(
                this.swapchain,
                0,
                &mut image_count,
                ptr::null_mut(),
            ))?
        };

        let mut images = vec![
            sys::SwapchainImageOpenGLESKHR {
                ty: sys::SwapchainImageOpenGLESKHR::TYPE,
                next: ptr::null_mut(),
                image: 0,
            };
            image_count as usize
        ];
        unsafe {
            super::xr_res((ext_fns.enumerate_environment_depth_swapchain_images)(
                this.swapchain,
                image_count,
                &mut image_count,
                images.as_mut_ptr().cast(),
            ))?
        };
        images.truncate(image_count as usize);
        this.swapchain_images = images.iter().map(|image| image.image).collect();

        unsafe { super::xr_res((ext_fns.start_environment_depth_provider)(provider))? };

        Ok(this)
    }

    pub fn swapchain_images(&self) -> &[u32] {
        &self.swapchain_images
    }

    pub fn resolution(&self) -> (u32, u32) {
        self.resolution
    }

    // Returns None while no depth image is available yet
    pub fn acquire_image(
        &self,
        space: &xr::Space,
        display_time: xr::Time,
    ) -> xr::Result<Option<EnvironmentDepthImageMETA>> {
        let acquire_info = sys::EnvironmentDepthImageAcquireInfoMETA {
            ty: sys::EnvironmentDepthImageAcquireInfoMETA::TYPE,
            next: ptr::null(),
            space: space.as_raw(),
            display_time,
        };

        let default_view = sys::EnvironmentDepthImageViewMETA {
            ty: sys::EnvironmentDepthImageViewMETA::TYPE,
            next: ptr::null(),
            fov: xr::Fovf::default(),
            pose: xr::Posef::IDENTITY,
        };
        let mut image = sys::EnvironmentDepthImageMETA {
            ty: sys::EnvironmentDepthImageMETA::TYPE,
            next: ptr::null(),
            swapchain_index: 0,
            near_z: 0.0,
            far_z: 0.0,
            views: [default_view; 2],
        };

        let result = unsafe {
            (self.ext_fns.acquire_environment_depth_image)(self.provider, &acquire_info, &mut image)
        };

        if result == sys::Result::ENVIRONMENT_DEPTH_NOT_AVAILABLE_META {
            return Ok(None);
        }
        super::xr_res(result)?;

        Ok(Some(EnvironmentDepthImageMETA {
            swapchain_index: image.swapchain_index,
            near_z: image.near_z,
            far_z: image.far_z,
            views: image.views.map(|view| (view.pose, view.fov)),
        }))
    }
}

impl Drop for EnvironmentDepthMETA {
    fn drop(&mut self) {
        unsafe {
            (self.ext_fns.stop_environment_depth_provider)(self.provider);
            if self.swapchain != sys::EnvironmentDepthSwapchainMETA::NULL {
                (self.ext_fns.destroy_environment_depth_swapchain)(self.swapchain);
            }
            (self.ext_fns.destroy_environment_depth_provider)(self.provider);
        }
    }
}
