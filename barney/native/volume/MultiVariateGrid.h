// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/ModelSlot.h"
#include "native/volume/MCAccelerator.h"

/*! \file MultiVariateGrid.h multi-variate fields on a dense regular grid

  A multi-variate grid is ONE scalar field whose voxels each carry several
  co-located variables ("channels"), each a scalar or a 3-vector. Before
  anything is classified, the channels are mixed into one value per voxel and
  then reduced to the single scalar that the volume's transfer function sees:

      v(voxel) = sum_c channelWeights[c] * channel_c(voxel)    (mix, linear)
      s(P)     = reduce( trilinear(v)(P) )                     (reduction)

  where reduce is one of: magnitude |v|, component v[k], or dot(v,direction).

  This is a different paradigm from several volumes overlapping in space.
  Those are separate Volume objects, each with its own field and transfer
  function, composited by the renderer. Here there is one field and one
  transfer function, and the variables interact before classification
  (|a+b| != |a|+|b|), which is what e.g. per-orbital current channels need.

  The mix runs once per voxel whenever the weights change, never per sample:
  it is linear, so it commutes with trilinear interpolation, and a sample
  costs one interpolated fetch however many channels there are. With
  BARNEY_MULTIVARIATE_HW_INTERP on a CUDA/OptiX backend the mixed grid is a
  CUDA array that the mix kernel writes through a surface object and the
  sampler reads through a hardware-filtered texture; otherwise it is a linear
  buffer interpolated in software.

  Majorants: trilinear interpolation is a convex combination of the 8 cell
  corners, so a linear reduction (component, dot) is bounded by its corner
  min/max, and |v| by its corner max, with 0 as the lower end because |v| can
  dip below every corner inside a cell.

  Parameters:
  - "dims" (int3), "gridOrigin" (float3), "gridSpacing" (float3)
  - "numChannels" (int), "numComponents" (int, 1 or 3)
  - "channels" (float data, numChannels*numVoxels*numComponents, channel-major,
    x fastest)
  - "channelWeights" (float data, numChannels; all 1 if not set)
  - "reduction" ("magnitude" | "component" | "dot", or 0/1/2)
  - "component" (int), "direction" (float3)
*/

#if BARNEY_MULTIVARIATE_HW_INTERP && (BARNEY_RTC_OPTIX || BARNEY_RTC_CUDA) \
  && !defined(__HIPCC__) && !defined(__HIP_PLATFORM_AMD__) && !defined(USE_HIP)
# define BARNEY_MV_TEXTURE 1
#else
# define BARNEY_MV_TEXTURE 0
#endif

namespace BARNEY_NS {
  namespace native {

    struct MultiVariateGridField : public ScalarField
    {
      typedef std::shared_ptr<MultiVariateGridField> SP;

      typedef enum { MAGNITUDE=0, COMPONENT, DOT } Reduction;

      /*! the mixed grid and how to reduce it; the channels and weights
          themselves are only needed by the mix kernel */
      struct DD : public ScalarField::DD {
#if BARNEY_MV_TEXTURE
        rtc::TextureObject mixedTex;
#else
        const float       *mixed;
#endif
        vec3i dims;
        vec3f gridOrigin;
        vec3f gridSpacing;
        int   numComponents;
        int   reduction;
        int   component;
        vec3f direction;

#if RTC_DEVICE_CODE
        inline __rtc_device float reduce(const vec3f v) const;
        inline __rtc_device vec3f fetch(const vec3f rel) const;
#endif
      };

      MultiVariateGridField(Context *context, const DevGroup::SP &devices);
      virtual ~MultiVariateGridField();

      // ------------------------------------------------------------------
      bool set1i(const std::string &member, const int &value) override;
      bool set3i(const std::string &member, const vec3i &value) override;
      bool set3f(const std::string &member, const vec3f &value) override;
      bool setString(const std::string &member,
                     const std::string &value) override;
      bool setData(const std::string &member,
                   const std::shared_ptr<Data> &value) override;
      bool setObject(const std::string &member,
                     const Object::SP &value) override;
      void commit() override;
      // ------------------------------------------------------------------

      DD getDD(Device *device);
      MCGrid::SP buildMCs() override;
      VolumeAccel::SP createAccel(Volume *volume) override;
      IsoSurfaceAccel::SP createIsoAccel(IsoSurface *isoSurface) override;

      static std::string typeName() { return "MultiVariateGrid"; }

      /*! re-mix the channels, then rebuild the macro-cell ranges and the
          majorants of every volume built on this field */
      void refresh();

      PODData::SP channels;
      PODData::SP channelWeights;
      vec3i dims          { 0,0,0 };
      vec3f gridOrigin    { 0,0,0 };
      vec3f gridSpacing   { 1,1,1 };
      int   numChannels   = 0;
      int   numComponents = 1;
      int   reduction     = MAGNITUDE;
      int   component     = 0;
      vec3f direction     { 1,0,0 };
      bool  dirty         = true;

      /*! a weight change commits the field, not the volume, so the field
          pushes new majorants to its accels itself */
      std::vector<std::weak_ptr<VolumeAccel>> accels;

      struct PLD {
#if BARNEY_MV_TEXTURE
        cudaArray_t         array    = 0;
        cudaTextureObject_t tex      = 0;
        cudaTextureObject_t texPoint = 0;
        cudaSurfaceObject_t surf     = 0;
#else
        float *values = 0;
#endif
        vec3i allocDims       { 0,0,0 };
        int   allocComponents = 0;
      };
      PLD *getPLD(Device *device);
      std::vector<PLD> perLogical;

    private:
      bool valid() const;
      void allocMixed(Device *device);
      void freeMixed(Device *device);
      bool setArray(const std::string &member, const PODData::SP &data);
    };

    /*! sampler for a MultiVariateGridField; a regular grid needs no
        acceleration structure, so build() is a no-op */
    struct MultiVariateGridSampler : public ScalarFieldSampler {
      MultiVariateGridSampler(MultiVariateGridField *const sf) : sf(sf) {}

      struct DD : public MultiVariateGridField::DD {
#if RTC_DEVICE_CODE
        inline __rtc_device float sample(const vec3f P, bool dbg=false) const;
#endif
      };

      void build() override {}
      DD getDD(Device *device);
      inline static std::string typeName() { return "MultiVariateGrid"; }
      MultiVariateGridField *const sf;
    };

#if RTC_DEVICE_CODE
    inline __rtc_device
    float MultiVariateGridField::DD::reduce(const vec3f v) const
    {
      if (reduction == COMPONENT)
        return component == 0 ? v.x : (component == 1 ? v.y : v.z);
      if (reduction == DOT)
        return dot(v,direction);
      return sqrtf(dot(v,v));
    }

    /*! the mixed value at grid-relative position rel, which must lie inside
        [0,dims-1]; scalar grids return it in .x */
    inline __rtc_device
    vec3f MultiVariateGridField::DD::fetch(const vec3f rel) const
    {
#if BARNEY_MV_TEXTURE
      if (numComponents == 1)
        return vec3f(rtc::tex3D<float>(mixedTex,rel.x+.5f,rel.y+.5f,rel.z+.5f),
                     0.f,0.f);
      const vec4f t = rtc::tex3D<vec4f>(mixedTex,rel.x+.5f,rel.y+.5f,rel.z+.5f);
      return vec3f(t.x,t.y,t.z);
#else
      const vec3i i0 = vec3i(int(rel.x),int(rel.y),int(rel.z));
      const vec3i i1 = min(i0+vec3i(1),dims-vec3i(1));
      const vec3f f  = rel - vec3f(i0);
      vec3f v(0.f);
#pragma unroll
      for (int k=0;k<8;k++) {
        const int x = (k&1) ? i1.x : i0.x;
        const int y = (k&2) ? i1.y : i0.y;
        const int z = (k&4) ? i1.z : i0.z;
        const float w
          = ((k&1) ? f.x : 1.f-f.x)
          * ((k&2) ? f.y : 1.f-f.y)
          * ((k&4) ? f.z : 1.f-f.z);
        const float *c
          = mixed + (x + (size_t)dims.x*(y + (size_t)dims.y*z))*numComponents;
        v = v + w * (numComponents == 1 ? vec3f(c[0],0.f,0.f)
                                        : vec3f(c[0],c[1],c[2]));
      }
      return v;
#endif
    }

    inline __rtc_device
    float MultiVariateGridSampler::DD::sample(const vec3f P, bool dbg) const
    {
      const vec3f rel = (P - gridOrigin) * rcp(gridSpacing);
      if (rel.x < 0.f || rel.y < 0.f || rel.z < 0.f) return NAN;
      if (rel.x > dims.x-1.f || rel.y > dims.y-1.f || rel.z > dims.z-1.f)
        return NAN;
      return reduce(fetch(rel));
    }
#endif
  }
}
