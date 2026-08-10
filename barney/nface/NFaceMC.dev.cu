// SPDX-FileCopyrightText:
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier:
// Apache-2.0


/*! \file NFaceMC.dev.cu implements a macro-cell accelerated
    shared-face polyhedral ("nface") mesh data type.

    This particular volume type:

    - uses cubql to accelerate point-in-cell queries (for the scalar
      field evaluation)

    - uses macro cells and DDA traversal for domain traversal
*/

#include "barney/nface/NFaceCuBQLSampler.h"
#include "barney/volume/DDA.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {

  struct NFaceMC_Programs {

    static inline __rtc_device
    void bounds(const rtc::TraceInterface &ti,
                const void *geomData,
                owl::common::box3f &bounds,
                const int32_t primID)
    {
#if RTC_DEVICE_CODE
      MCVolumeAccel<NFaceCuBQLSampler>::boundsProg(ti,geomData,bounds,primID);
#endif
    }

    static inline __rtc_device
    void intersect(rtc::TraceInterface &ti)
    {
#if RTC_DEVICE_CODE
      MCVolumeAccel<NFaceCuBQLSampler>::isProg(ti);
#endif
    }

    static inline __rtc_device
    void closestHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }

    static inline __rtc_device
    void anyHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }
  };


  struct MCIsoAccel_NFace_Programs {
    static inline __rtc_device
    void bounds(const rtc::TraceInterface &ti,
                const void *geomData,
                owl::common::box3f &bounds,
                const int32_t primID)
    {
#if RTC_DEVICE_CODE
      MCIsoSurfaceAccel<NFaceCuBQLSampler>
        ::boundsProg(ti,geomData,bounds,primID);
#endif
    }

    static inline __rtc_device
    void intersect(rtc::TraceInterface &ti)
    {
#if RTC_DEVICE_CODE
      MCIsoSurfaceAccel<NFaceCuBQLSampler>
        ::isProg(ti);
#endif
    }

    static inline __rtc_device
    void closestHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }

    static inline __rtc_device
    void anyHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }
  };



  using NFaceMC = MCVolumeAccel<NFaceCuBQLSampler>;
  using NFaceMC_Iso = MCIsoSurfaceAccel<NFaceCuBQLSampler>;

  RTC_EXPORT_USER_GEOM(NFaceMC,NFaceMC::DD,NFaceMC_Programs,false,false);
  RTC_EXPORT_USER_GEOM(NFaceMC_Iso,NFaceMC_Iso::DD,
                       MCIsoAccel_NFace_Programs,false,false);
}
