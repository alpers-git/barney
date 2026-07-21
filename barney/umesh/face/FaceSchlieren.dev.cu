// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

/*! \file FaceSchlieren.dev.cu

    Face-iteration schlieren integrator for tet meshes. The element faces are
    a triangle BVH; an any-hit at each crossing adds the order-independent term
    sign(d.N) * (g_minus - g_plus) * t to the ray's accumulated gradient
    integral, then ignores the intersection so the ray keeps traversing. Summed
    over all crossings this is exactly the integral of the per-element density
    gradient along the ray, with no front-to-back ordering required.
*/

#include "barney/umesh/face/FaceSchlierenAccel.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {
  using namespace BARNEY_NS::render;

  struct FaceSchlierenPrograms {
#if RTC_DEVICE_CODE
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }

    static inline __rtc_device
    void anyHit(rtc::TraceInterface &ti)
    {
      Ray &ray = *(Ray *)ti.getPRD();
      const FaceSchlierenAccel::DD &self
        = *(const FaceSchlierenAccel::DD *)ti.getProgramData();

      const int primID = ti.getPrimitiveIndex();
      const vec3i tri = self.indices[primID];
      const vec3f v0 = self.vertices[tri.x];
      const vec3f v1 = self.vertices[tri.y];
      const vec3f v2 = self.vertices[tri.z];
      const vec3f N = cross(v1 - v0, v2 - v0);
      const vec3f d = ti.getObjectRayDirection();
      const float t = ti.getRayTmax();

      const vec2i cells = self.faceCells[primID];
      const vec3f gMinus = (cells.x >= 0) ? self.cellGradients[cells.x] : vec3f(0.f);
      const vec3f gPlus  = (cells.y >= 0) ? self.cellGradients[cells.y] : vec3f(0.f);
      const float s = (dot(d, N) > 0.f) ? 1.f : -1.f;

      ray.schlieren = ray.schlieren + s * (gMinus - gPlus) * t;
      ray.isSchlieren = 1;
      ti.ignoreIntersection();
    }
#endif
  };

  RTC_EXPORT_TRIANGLES_GEOM(FaceSchlieren, FaceSchlierenAccel::DD,
                            FaceSchlierenPrograms, true, false);
}
