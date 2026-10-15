// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/ModelSlot.h"
#include "native/volume/MCAccelerator.h"

namespace BARNEY_NS {
  namespace native {

    /*! an orbital-current field on a dense regular grid.

      Same physics and the same filtering as the unstructured (.orbm) version:
      every voxel carries one current *vector* per orbital channel, the enabled
      channels are summed per voxel into 'activeVec', that one vector is
      interpolated at the sample point, and its magnitude is what reaches the
      transfer function. Vectors rather than magnitudes because
      |sum_c j_c| != sum_c |j_c| wherever channels oppose.

      What differs is the container. The tet version stores a uniform lattice
      cut into 6 tets per cube, which costs 567MB of connectivity and a cuBQL
      bvh over 37M primitives to answer a question - "which element contains
      P?" - that on a regular grid is three divisions. Sampling here is a
      trilinear fetch with no acceleration structure at all.

      Majorants work out exactly as they do for the tets, and for the same
      reason: trilinear interpolation is a convex combination of the 8 corner
      vectors, so |sum_i w_i a_i| <= max_i |a_i|. The maximum of |j| over a
      cell is therefore bounded by the corner values, and a macro cell's range
      is [0, max |activeVec| over the voxels it covers]. The low end is 0
      rather than the corner minimum because |j| is convex and can dip below
      every corner inside the cell. */
    struct OrbGridField : public ScalarField
    {
      typedef std::shared_ptr<OrbGridField> SP;

      OrbGridField(Context *context, const DevGroup::SP &devices);
      virtual ~OrbGridField() = default;

      struct DD : public ScalarField::DD {
        /*! numChannels * nx*ny*nz vec3f, channel-major */
        const vec3f *channelVec;
        /*! per-voxel sum over the enabled channels */
        const vec3f *activeVec;
        vec3i        dims;
        vec3f        gridOrigin;
        vec3f        gridSpacing;
        int          numChannels;
        uint32_t     channelMask;

        inline __rtc_device int voxelIndex(vec3i c) const
        { return c.x + dims.x*(c.y + dims.y*c.z); }
      };

      // ------------------------------------------------------------------
      bool set3i(const std::string &member, const vec3i &value) override;
      bool set3f(const std::string &member, const vec3f &value) override;
      bool set1i(const std::string &member, const int &value) override;
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

      static std::string typeName() { return "OrbGrid"; }

      /*! set the enabled-channel mask; rebuilds the per-voxel active vectors,
        the macro-cell ranges and the majorants derived from them */
      void setChannelMask(uint32_t mask);
      void refreshVecMCs();

      PODData::SP channelVecData;
      vec3i    dims        { 0,0,0 };
      vec3f    gridOrigin  { 0,0,0 };
      vec3f    gridSpacing { 1,1,1 };
      int      numChannels = 0;
      uint32_t channelMask = ~0u;

      /*! a mask change commits the field, not the volume, so the field has to
        push the new majorants to its accels itself */
      std::vector<std::weak_ptr<VolumeAccel>> accels;

      struct PLD { vec3f *activeVec = 0; };
      PLD *getPLD(Device *device);
      std::vector<PLD> perLogical;
    };

    /*! sampler for an OrbGridField. build() is a no-op: a regular grid needs
      no acceleration structure to locate the cell containing a point. */
    struct OrbGridSampler : public ScalarFieldSampler {
      OrbGridSampler(OrbGridField *const sf) : sf(sf) {}

      struct DD : public OrbGridField::DD {
#if RTC_DEVICE_CODE
        inline __rtc_device float sample(const vec3f P, bool dbg=false) const;
#endif
      };

      void build() override {}
      DD getDD(Device *device);
      inline static std::string typeName() { return "OrbGrid"; }
      OrbGridField *const sf;
    };

#if RTC_DEVICE_CODE
    /*! |sum_{c enabled} j_c| at P.

      The channels were already summed per voxel into activeVec, which is exact
      rather than an approximation: interpolation is linear, so summing at the
      voxels and interpolating equals interpolating each channel and summing at
      P. The per-sample cost is therefore one trilinear fetch however many
      channels are switched on. The magnitude is taken once, here, after the
      vectors are combined - never per channel, and never after classification. */
    inline __rtc_device
    float OrbGridSampler::DD::sample(const vec3f P, bool dbg) const
    {
      vec3f rel = (P - gridOrigin) * rcp(gridSpacing);
      if (rel.x < 0.f || rel.y < 0.f || rel.z < 0.f) return 0.f;
      if (rel.x > dims.x-1.f || rel.y > dims.y-1.f || rel.z > dims.z-1.f)
        return 0.f;
      const vec3i i0 = vec3i(int(rel.x),int(rel.y),int(rel.z));
      const vec3i i1 = vec3i(min(i0.x+1,dims.x-1),
                             min(i0.y+1,dims.y-1),
                             min(i0.z+1,dims.z-1));
      const vec3f f = rel - vec3f(float(i0.x),float(i0.y),float(i0.z));
      vec3f j(0.f,0.f,0.f);
#pragma unroll
      for (int k=0;k<8;k++) {
        const int cx = (k&1) ? i1.x : i0.x;
        const int cy = (k&2) ? i1.y : i0.y;
        const int cz = (k&4) ? i1.z : i0.z;
        const float w
          = ((k&1) ? f.x : 1.f-f.x)
          * ((k&2) ? f.y : 1.f-f.y)
          * ((k&4) ? f.z : 1.f-f.z);
        j = j + w * activeVec[voxelIndex(vec3i(cx,cy,cz))];
      }
      return sqrtf(j.x*j.x + j.y*j.y + j.z*j.z);
    }
#endif
  }
}
