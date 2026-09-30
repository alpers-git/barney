// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file ParticleRBFMC.dev.cu implements a macro-cell accelerated field of
  signed radial basis functions.

  This particular volume type:

  - uses cubql range queries to sum all basis functions covering a sample point

  - uses macro cells and DDA traversal for domain traversal
*/

#include "native/volume/ParticleRBFSampler.h"
#include "native/volume/DDA.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::native::OptixGlobals);

namespace BARNEY_NS {
  namespace native {

    struct ParticleRBFMC_Programs {
      static inline __rtc_device
      void bounds(const rtc::TraceInterface &ti,
                  const void *geomData,
                  owl::common::box3f &bounds,
                  const int32_t primID)
      {
#if RTC_DEVICE_CODE
        MCVolumeAccel<ParticleRBFSampler>::boundsProg(ti,geomData,bounds,primID);
#endif
      }

      static inline __rtc_device
      void intersect(rtc::TraceInterface &ti)
      {
#if RTC_DEVICE_CODE
        MCVolumeAccel<ParticleRBFSampler>::isProg(ti);
#endif
      }

      static inline __rtc_device
      void closestHit(rtc::TraceInterface &ti)
      { /* nothing to do */ }

      static inline __rtc_device
      void anyHit(rtc::TraceInterface &ti)
      { /* nothing to do */ }
    };

    using ParticleRBFMC = MCVolumeAccel<ParticleRBFSampler>;

    RTC_EXPORT_USER_GEOM(ParticleRBFMC,
                         ParticleRBFMC::DD,
                         ParticleRBFMC_Programs,
                         false,false);
  }
}
