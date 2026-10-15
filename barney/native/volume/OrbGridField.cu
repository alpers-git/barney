// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/common/barney-common.h"
#include "native/volume/OrbGridField.h"
#include "native/Context.h"
#include "native/volume/MCGrid.cuh"
#include <chrono>

namespace BARNEY_NS {
  namespace native {

    RTC_IMPORT_USER_GEOM(/*file*/OrbGridMC,/*name*/OrbGridMC,
                         MCVolumeAccel<OrbGridSampler>::DD,false,false);

    OrbGridField::OrbGridField(Context *context, const DevGroup::SP &devices)
      : ScalarField(context,devices)
    { perLogical.resize(devices->numLogical); }

    OrbGridField::PLD *OrbGridField::getPLD(Device *device)
    {
      assert(device && device->contextRank() >= 0);
      assert(device->contextRank() < (int)perLogical.size());
      return &perLogical[device->contextRank()];
    }

    /*! activeVec[v] = sum over the ENABLED channels of that channel's vector */
    __rtc_global void orbGridRecombine(rtc::ComputeInterface ci,
                                       OrbGridField::DD field,
                                       vec3f *activeVec,
                                       int numVoxels)
    {
#if RTC_DEVICE_CODE
      const int v = ci.launchIndex().x;
      if (v >= numVoxels) return;
      vec3f sum(0.f,0.f,0.f);
      for (int c=0;c<field.numChannels;c++)
        if (field.channelMask & (1u<<c))
          sum = sum + field.channelVec[(size_t)c*numVoxels + v];
      activeVec[v] = sum;
#endif
    }

    /*! one thread per macro cell: gather the largest |activeVec| over the
      voxels that cell covers, plus one voxel of overlap on each side so a
      cell boundary falling inside a grid cell is still bounded.

      [0,max] rather than [min,max]: trilinear interpolation makes |j| convex
      inside a grid cell, so its maximum is attained at a corner - that is what
      makes this bound exact - but its minimum can dip below every corner, and
      a range starting at the corner minimum would be too narrow for a transfer
      function whose opacity is not monotonic. */
    __rtc_global void orbGridComputeMCs(rtc::ComputeInterface ci,
                                        OrbGridField::DD field,
                                        MCGrid::DD grid,
                                        vec3i cellsPerMC)
    {
#if RTC_DEVICE_CODE
      const int mcIdx = ci.launchIndex().x;
      const int numMC = grid.dims.x*grid.dims.y*grid.dims.z;
      if (mcIdx >= numMC) return;
      const vec3i mc = grid.cellID(mcIdx);

      const vec3i lo = max(vec3i(0,0,0),
                           vec3i(mc.x*cellsPerMC.x-1,
                                 mc.y*cellsPerMC.y-1,
                                 mc.z*cellsPerMC.z-1));
      const vec3i hi = min(field.dims-vec3i(1,1,1),
                           vec3i((mc.x+1)*cellsPerMC.x+1,
                                 (mc.y+1)*cellsPerMC.y+1,
                                 (mc.z+1)*cellsPerMC.z+1));
      float maxMag = 0.f;
      bool any = false;
      for (int z=lo.z;z<=hi.z;z++)
        for (int y=lo.y;y<=hi.y;y++)
          for (int x=lo.x;x<=hi.x;x++) {
            const vec3f j = field.activeVec[field.voxelIndex(vec3i(x,y,z))];
            const float m = sqrtf(j.x*j.x+j.y*j.y+j.z*j.z);
            if (m > maxMag) maxMag = m;
            any = true;
          }
      range1f r;
      if (any) { r.lower = 0.f; r.upper = maxMag; }
      grid.scalarRanges[mcIdx] = r;
#endif
    }

    /*! how many grid cells one macro cell spans. Finer macro cells give
      tighter majorants and fewer woodcock steps, but cost more to rebuild on
      every channel toggle; 8 keeps a rebuild in the low milliseconds. */
    static vec3i cellsPerMC() { return vec3i(4,4,4); }

    void OrbGridField::refreshVecMCs()
    {
      if (numChannels <= 0) return;
      const auto t0 = std::chrono::steady_clock::now();
      const int numVoxels = dims.x*dims.y*dims.z;
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto rtc = device->rtc;
        PLD *pld = getPLD(device);
        if (!pld->activeVec)
          pld->activeVec = (vec3f*)rtc->allocMem((size_t)numVoxels*sizeof(vec3f));
        __rtc_launch(rtc,orbGridRecombine,
                     divRoundUp(numVoxels,256),256,
                     getDD(device),pld->activeVec,numVoxels);
        rtc->sync();
      }
      if (!mcGrid || !mcGrid->built()) return;
      const int numMC = mcGrid->dims.x*mcGrid->dims.y*mcGrid->dims.z;
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto rtc = device->rtc;
        __rtc_launch(rtc,orbGridComputeMCs,
                     divRoundUp(numMC,128),128,
                     getDD(device),mcGrid->getDD(device),cellsPerMC());
        rtc->sync();
      }
      /* the ranges changed without the transfer function changing, so nothing
         downstream would know to recompute the majorants; a stale majorant is
         too small wherever the new range is larger and woodcock tracking then
         misses extinction. */
      ++mcGrid->contentEpoch;
      for (auto &weak : accels)
        if (auto accel = weak.lock())
          accel->rebuildMajorantsOnly();
      const double ms = std::chrono::duration<double,std::milli>
        (std::chrono::steady_clock::now()-t0).count();
      std::cout << "#bn.og: channel refresh " << ms << " ms" << std::endl;
    }

    void OrbGridField::setChannelMask(uint32_t mask)
    {
      if (numChannels <= 0 || mask == channelMask) return;
      channelMask = mask;
      refreshVecMCs();
    }

    MCGrid::SP OrbGridField::buildMCs()
    {
      MCGrid::SP grid = std::make_shared<MCGrid>(devices);
      const vec3i cpm = cellsPerMC();
      grid->resize(divRoundUp(dims-vec3i(1,1,1),cpm));
      grid->gridOrigin  = gridOrigin;
      grid->gridSpacing = vec3f(float(cpm.x),float(cpm.y),float(cpm.z))*gridSpacing;
      std::cout << OWL_TERMINAL_BLUE
                << "#bn.og: macro cell grid of " << grid->dims
                << " over a " << dims << " voxel grid"
                << OWL_TERMINAL_DEFAULT << std::endl;
      mcGrid = grid;
      refreshVecMCs();
      return grid;
    }

    OrbGridField::DD OrbGridField::getDD(Device *device)
    {
      DD dd;
      (ScalarField::DD &)dd = ScalarField::getDD(device);
      dd.channelVec  = channelVecData
        ? (const vec3f*)channelVecData->getDD(device) : nullptr;
      dd.activeVec   = (const vec3f*)getPLD(device)->activeVec;
      dd.dims        = dims;
      dd.gridOrigin  = gridOrigin;
      dd.gridSpacing = gridSpacing;
      dd.numChannels = numChannels;
      dd.channelMask = channelMask;
      return dd;
    }

    OrbGridSampler::DD OrbGridSampler::getDD(Device *device)
    {
      DD dd;
      (OrbGridField::DD &)dd = sf->getDD(device);
      return dd;
    }

    VolumeAccel::SP OrbGridField::createAccel(Volume *volume)
    {
      auto sampler = std::make_shared<OrbGridSampler>(this);
      auto accel = std::make_shared<MCVolumeAccel<OrbGridSampler>>
        (volume,createGeomType_OrbGridMC,sampler);
      accels.push_back(accel);
      return accel;
    }

    IsoSurfaceAccel::SP OrbGridField::createIsoAccel(IsoSurface *isoSurface)
    { return {}; }

    bool OrbGridField::set3i(const std::string &member, const vec3i &value)
    {
      if (member == "dims") { dims = value; return true; }
      return ScalarField::set3i(member,value);
    }

    bool OrbGridField::set3f(const std::string &member, const vec3f &value)
    {
      if (member == "gridOrigin")  { gridOrigin = value;  return true; }
      if (member == "gridSpacing") { gridSpacing = value; return true; }
      return ScalarField::set3f(member,value);
    }

    bool OrbGridField::set1i(const std::string &member, const int &value)
    {
      if (member == "numChannels") { numChannels = value; return true; }
      if (member == "channelMask") {
        /* the generic custom-field path re-applies every parameter in name
           order, so the mask can arrive before numChannels does. Store it
           either way; commit() builds the macro cells once the rest is in. */
        if (numChannels > 0) setChannelMask((uint32_t)value);
        else                 channelMask = (uint32_t)value;
        return true;
      }
      return ScalarField::set1i(member,value);
    }

    bool OrbGridField::setData(const std::string &member,
                               const std::shared_ptr<Data> &value)
    {
      if (ScalarField::setData(member,value)) return true;
      if (member == "channel.vector") {
        channelVecData = value->as<PODData>();
        return true;
      }
      return false;
    }

    /*! the generic custom-field bridge hands arrays over as objects rather
      than through setData, so accept them here too */
    bool OrbGridField::setObject(const std::string &member,
                                 const Object::SP &value)
    {
      if (member == "channel.vector") {
        if (auto pod = std::dynamic_pointer_cast<PODData>(value)) {
          channelVecData = pod;
          return true;
        }
      }
      return ScalarField::setObject(member,value);
    }

    void OrbGridField::commit()
    {
      assert(channelVecData);
      assert(dims.x>0 && dims.y>0 && dims.z>0);
      worldBounds.lower = gridOrigin;
      worldBounds.upper = gridOrigin + gridSpacing*vec3f(float(dims.x-1),
                                                         float(dims.y-1),
                                                         float(dims.z-1));
    }
  }
}
