// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file MultiVariateGridMC.dev.cu macro-cell accelerated multi-variate grid. */

#include "native/volume/MultiVariateGrid.h"
#include "native/volume/DDA.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::native::OptixGlobals);

namespace BARNEY_NS {
  namespace native {

    struct MultiVariateGridMC_Programs {
      static inline __rtc_device
      void bounds(const rtc::TraceInterface &ti, const void *geomData,
                  owl::common::box3f &bounds, const int32_t primID)
      {
#if RTC_DEVICE_CODE
        MCVolumeAccel<MultiVariateGridSampler>::boundsProg(ti,geomData,bounds,primID);
#endif
      }
      static inline __rtc_device
      void intersect(rtc::TraceInterface &ti)
      {
#if RTC_DEVICE_CODE
        MCVolumeAccel<MultiVariateGridSampler>::isProg(ti);
#endif
      }
      static inline __rtc_device void closestHit(rtc::TraceInterface &ti) {}
      static inline __rtc_device void anyHit(rtc::TraceInterface &ti) {}
    };

    using MultiVariateGridMC = MCVolumeAccel<MultiVariateGridSampler>;
    RTC_EXPORT_USER_GEOM(MultiVariateGridMC,MultiVariateGridMC::DD,
                         MultiVariateGridMC_Programs,false,false);
  }
}
