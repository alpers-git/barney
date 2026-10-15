// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file OrbGridMC.dev.cu macro-cell accelerated orbital-current grid. */

#include "native/volume/OrbGridField.h"
#include "native/volume/DDA.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::native::OptixGlobals);

namespace BARNEY_NS {
  namespace native {

    struct OrbGridMC_Programs {
      static inline __rtc_device
      void bounds(const rtc::TraceInterface &ti, const void *geomData,
                  owl::common::box3f &bounds, const int32_t primID)
      {
#if RTC_DEVICE_CODE
        MCVolumeAccel<OrbGridSampler>::boundsProg(ti,geomData,bounds,primID);
#endif
      }
      static inline __rtc_device
      void intersect(rtc::TraceInterface &ti)
      {
#if RTC_DEVICE_CODE
        MCVolumeAccel<OrbGridSampler>::isProg(ti);
#endif
      }
      static inline __rtc_device void closestHit(rtc::TraceInterface &ti) {}
      static inline __rtc_device void anyHit(rtc::TraceInterface &ti) {}
    };

    using OrbGridMC = MCVolumeAccel<OrbGridSampler>;
    RTC_EXPORT_USER_GEOM(OrbGridMC,OrbGridMC::DD,OrbGridMC_Programs,false,false);
  }
}
