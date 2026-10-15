// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/common/barney-common.h"
#include "native/umesh/common/UMeshField.h"
#include "native/Context.h"
#include "native/umesh/mc/UMeshCuBQLSampler.h"
#include "native/volume/MCGrid.cuh"
#include <chrono>

namespace BARNEY_NS {
  namespace native {

    RTC_IMPORT_USER_GEOM(/*file*/UMeshMC,/*name*/UMeshMC,
                         /*geomtype device data */
                         MCVolumeAccel<UMeshCuBQLSampler>::DD,false,false);
    RTC_IMPORT_USER_GEOM(/*file*/UMeshMC,/*name*/UMeshMC_Iso,
                         /*geomtype device data */
                         MCIsoSurfaceAccel<UMeshCuBQLSampler>::DD,false,false);
  
    UMeshField::PLD *UMeshField::getPLD(Device *device) 
    {
      assert(device);
      assert(device->contextRank() >= 0);
      assert(device->contextRank() < perLogical.size());
      return &perLogical[device->contextRank()];
    }

#if RTC_DEVICE_CODE
    inline __rtc_device float length3(vec4f v)
    { return length(getPos(v)); }
  
    template<int D> inline __rtc_device
    void rasterTet(MCGrid::DD grid,
                   vec4f a,
                   vec4f b,
                   vec4f c,
                   vec4f d)
    {
      float lab = length3(b-a);
      float lac = length3(c-a);
      float lad = length3(d-a);
      float lbc = length3(c-b);
      float lbd = length3(d-b);
      float lcd = length3(d-c);
      float maxLen = max(max(max(max(max(lab,lac),lad),lbc),lbd),lcd);

      vec4f ab = 0.5f*(a+b);
      vec4f ac = 0.5f*(a+c);
      vec4f ad = 0.5f*(a+d);
      vec4f bc = 0.5f*(b+c);
      vec4f bd = 0.5f*(b+d);
      vec4f cd = 0.5f*(c+d);

      vec4f oa,ob,oc,od0,od1;
      if (lab >= maxLen) {
        oa = ab;
        ob = c;
        oc = d;
        od0 = a;
        od1 = b;
      } else if (lac >= maxLen) {
        oa = ac;
        ob = b;
        oc = d;
        od0 = a;
        od1 = c;
      } else if (lad >= maxLen) {
        oa = ad;
        ob = b;
        oc = c;
        od0 = a;
        od1 = d;
      } else if (lbc >= maxLen) {
        oa = bc;
        ob = a;
        oc = d;
        od0 = b;
        od1 = c;
      } else if (lbd >= maxLen) {
        oa = bd;
        ob = a;
        oc = c;
        od0 = b;
        od1 = d;
      } else {
        oa = cd;
        ob = a;
        oc = b;
        od0 = c;
        od1 = d;
      }
      rasterTet<D-1>(grid,oa,ob,oc,od0);
      rasterTet<D-1>(grid,oa,ob,oc,od1);
    }
  
    template<> inline __rtc_device
    void rasterTet<0>(MCGrid::DD grid,
                      vec4f a,
                      vec4f b,
                      vec4f c,
                      vec4f d)
    {
      box4f bb;
      bb.extend(a);
      bb.extend(b);
      bb.extend(c);
      bb.extend(d);
      rasterBox(grid,bb);
    }
#endif
  
    __rtc_global void umeshRasterCells(rtc::ComputeInterface ci,
                                       UMeshField::DD mesh,
                                       MCGrid::DD grid)
    {
#if RTC_DEVICE_CODE
      const int cellIdx = ci.launchIndex().x;
      if (cellIdx >= mesh.numCells) return;    

      auto cellType = mesh.cellTypes[cellIdx];
      if (cellType == _VTK_TET || cellType == _ANARI_TET) {
        const int ofs0 = mesh.cellOffsets[cellIdx];
        if (ofs0 < 0
            || (long long)ofs0 + 4 > (long long)mesh.numIndices)
          return;
        const int *ix = mesh.indices + ofs0;
        const vec4i id(ix[0], ix[1], ix[2], ix[3]);
        auto vtxOk = [&](int vi) {
          return vi >= 0 && vi < mesh.numVertices;
        };
        if (!vtxOk(id.x) || !vtxOk(id.y) || !vtxOk(id.z) || !vtxOk(id.w))
          return;
        int sidx_x = mesh.scalarsArePerVertex ? id.x : cellIdx;
        int sidx_y = mesh.scalarsArePerVertex ? id.y : cellIdx;
        int sidx_z = mesh.scalarsArePerVertex ? id.z : cellIdx;
        int sidx_w = mesh.scalarsArePerVertex ? id.w : cellIdx;
        if (mesh.scalarsArePerVertex) {
          if (sidx_x < 0 || sidx_x >= mesh.numScalars
              || sidx_y < 0 || sidx_y >= mesh.numScalars
              || sidx_z < 0 || sidx_z >= mesh.numScalars
              || sidx_w < 0 || sidx_w >= mesh.numScalars)
            return;
        } else if (cellIdx < 0 || cellIdx >= mesh.numScalars) {
          return;
        }
        vec4f a(mesh.vertices[id.x], mesh.scalars[sidx_x]);
        vec4f b(mesh.vertices[id.y], mesh.scalars[sidx_y]);
        vec4f c(mesh.vertices[id.z], mesh.scalars[sidx_z]);
        vec4f d(mesh.vertices[id.w], mesh.scalars[sidx_w]);
        rasterTet<5>(grid, a, b, c, d);
      } else {
        const box4f eltBounds = mesh.cellBounds(cellIdx);
        rasterBox(grid,getBox(mesh.worldBounds),eltBounds);
      }
#endif
    }
  
    // ==================================================================
    // orbital-channel vector field
    // ==================================================================

    /*! activeVec[v] = sum over the ENABLED channels of that channel's current
      vector at v.

      Summing here rather than at each sample is not an approximation:
      interpolation is linear, so sum_c interp(j_c)(P) == interp(sum_c j_c)(P)
      exactly. It makes the per-sample cost independent of the channel count,
      and it is what lets the macro-cell bound be computed from |activeVec|. */
    __rtc_global void umeshRecombineChannels(rtc::ComputeInterface ci,
                                             UMeshField::DD mesh,
                                             vec3f *activeVec)
    {
#if RTC_DEVICE_CODE
      const int v = ci.launchIndex().x;
      if (v >= mesh.numVertices) return;
      vec3f sum(0.f,0.f,0.f);
      for (int c=0;c<mesh.numChannels;c++)
        if (mesh.channelMask & (1u<<c))
          sum = sum + mesh.channelVec[(size_t)c*mesh.numVertices + v];
      activeVec[v] = sum;
#endif
    }

    /*! rasterize each tet's |j| range into the macro cells it overlaps.

      rasterTet subdivides and linearly interpolates the w component, so
      feeding it |activeVec| at the four vertices yields, at every subdivision
      point, the interpolated *magnitude* - which by the triangle inequality is
      >= the magnitude of the interpolated vector, the quantity actually
      sampled. The resulting upper bound is therefore conservative, never too
      small, which is what a majorant has to be.

      The low end is fixed up afterwards by umeshClampRangeLow: |j| is convex
      inside a tet, so its minimum can dip below the value at every vertex, and
      a range that starts at the vertex minimum would be too narrow for a
      transfer function whose opacity is not monotonic. */
    __rtc_global void umeshRasterVecCells(rtc::ComputeInterface ci,
                                          UMeshField::DD mesh,
                                          MCGrid::DD grid)
    {
#if RTC_DEVICE_CODE
      const int cellIdx = ci.launchIndex().x;
      if (cellIdx >= mesh.numCells) return;
      auto cellType = mesh.cellTypes[cellIdx];
      if (!(cellType == _VTK_TET || cellType == _ANARI_TET)) {
        const box4f eltBounds = mesh.cellBounds(cellIdx);
        rasterBox(grid,getBox(mesh.worldBounds),eltBounds);
        return;
      }
      const int ofs0 = mesh.cellOffsets[cellIdx];
      if (ofs0 < 0 || (long long)ofs0 + 4 > (long long)mesh.numIndices)
        return;
      const int *ix = mesh.indices + ofs0;
      const vec4i id(ix[0], ix[1], ix[2], ix[3]);
      auto vtxOk = [&](int vi) { return vi >= 0 && vi < mesh.numVertices; };
      if (!vtxOk(id.x) || !vtxOk(id.y) || !vtxOk(id.z) || !vtxOk(id.w))
        return;
      auto mag = [&](int vi) {
        vec3f j = mesh.activeVec[vi];
        return sqrtf(j.x*j.x + j.y*j.y + j.z*j.z);
      };
      vec4f a(mesh.vertices[id.x], mag(id.x));
      vec4f b(mesh.vertices[id.y], mag(id.y));
      vec4f c(mesh.vertices[id.z], mag(id.z));
      vec4f d(mesh.vertices[id.w], mag(id.w));
      /* depth 0, i.e. just this tet's bounding box.

         The scalar path uses depth 5 so that a tet spanning many macro cells
         gets a localised range in each of them. That does not apply here: this
         field's elements are finer than the macro cells, so a tet touches only
         one or two of them and subdividing gains essentially nothing - while
         costing 32 box rasterisations per tet, which over 37M tets is 1.2e9 of
         them on *every channel toggle*. Depth 0 is what makes a toggle
         interactive.

         It stays a valid majorant: |j| is convex inside a tet, so its maximum
         over the tet is attained at a vertex, and the box covers the tet. */
      rasterTet<0>(grid, a, b, c, d);
#endif
    }

    /*! pull every non-empty cell's range down to 0; see umeshRasterVecCells */
    __rtc_global void umeshClampRangeLow(rtc::ComputeInterface ci,
                                         MCGrid::DD grid,
                                         int numCells)
    {
#if RTC_DEVICE_CODE
      const int i = ci.launchIndex().x;
      if (i >= numCells) return;
      range1f r = grid.scalarRanges[i];
      if (r.lower > r.upper) return;           // still empty, leave it
      if (r.lower > 0.f) grid.scalarRanges[i].lower = 0.f;
#endif
    }

    void UMeshField::refreshVecMCs()
    {
      if (numChannels <= 0) return;
      const auto t0 = std::chrono::steady_clock::now();
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto rtc = device->rtc;
        PLD *pld = getPLD(device);
        if (!pld->activeVec)
          pld->activeVec
            = (vec3f *)rtc->allocMem((size_t)vertices->count*sizeof(vec3f));
        __rtc_launch(rtc,umeshRecombineChannels,
                     divRoundUp((int)vertices->count,256),256,
                     getDD(device),pld->activeVec);
        rtc->sync();
      }
      if (!mcGrid || !mcGrid->built()) return;
      const int numCells = mcGrid->dims.x*mcGrid->dims.y*mcGrid->dims.z;
      mcGrid->clearCells();
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto rtc = device->rtc;
        __rtc_launch(rtc,umeshRasterVecCells,
                     divRoundUp((int)cellOffsets->count,128),128,
                     getDD(device),mcGrid->getDD(device));
        __rtc_launch(rtc,umeshClampRangeLow,
                     divRoundUp(numCells,1024),1024,
                     mcGrid->getDD(device),numCells);
        rtc->sync();
      }
      /* the ranges just changed without the transfer function changing, so
         nothing downstream would know to recompute the majorants; a stale
         majorant is too small wherever the new range is larger, and woodcock
         tracking then misses extinction. */
      ++mcGrid->contentEpoch;
      for (auto &weak : accels)
        if (auto accel = weak.lock())
          accel->rebuildMajorantsOnly();
      const double ms = std::chrono::duration<double,std::milli>
        (std::chrono::steady_clock::now()-t0).count();
      std::cout << "#bn.um: channel refresh " << ms << " ms" << std::endl;
    }

    void UMeshField::setChannelMask(uint32_t mask)
    {
      if (numChannels <= 0 || mask == channelMask) return;
      channelMask = mask;
      refreshVecMCs();
    }

    MCGrid::SP UMeshField::buildMCs()
    {
      MCGrid::SP mcGrid = std::make_shared<MCGrid>(devices);
      buildInitialMacroCells(*mcGrid);
      return mcGrid;
    }
  
    /*! build *initial* macro-cell grid (ie, the scalar field min/max
      ranges, but not yet the majorants) over a umesh */
    void UMeshField::buildInitialMacroCells(MCGrid &grid)
    {
      if (grid.built()) {
        // initial grid already built
        return;
      }
      assert(!worldBounds.empty());

      float maxWidth = reduce_max(worldBounds.size());//getBox(worldBounds).size());
      int MC_GRID_SIZE
        = 400 + int(sqrtf(cellOffsets->count/1000.f));
      vec3i dims = 1+vec3i(worldBounds.size() * ((MC_GRID_SIZE-1) / maxWidth));
      std::cout << OWL_TERMINAL_BLUE
                << "#bn.um: building initial macro cell grid of " << dims << " MCs"
                << OWL_TERMINAL_DEFAULT << std::endl;
      grid.resize(dims);
    
      grid.gridOrigin  = worldBounds.lower;
      grid.gridSpacing = worldBounds.size() * rcp(vec3f(dims));
    
      grid.clearCells();
    
      if (numChannels > 0) {
        /* the ranges depend on which channels are enabled, so they are built
           by the same path that a mask change uses - activeVec first, then
           |activeVec| rasterised into the cells. */
        for (auto device : *devices) {
          SetActiveGPU forDuration(device);
          auto rtc = device->rtc;
          PLD *pld = getPLD(device);
          if (!pld->activeVec)
            pld->activeVec
              = (vec3f *)rtc->allocMem((size_t)vertices->count*sizeof(vec3f));
          __rtc_launch(rtc,umeshRecombineChannels,
                       divRoundUp((int)vertices->count,256),256,
                       getDD(device),pld->activeVec);
          rtc->sync();
          __rtc_launch(rtc,umeshRasterVecCells,
                       divRoundUp(numCells,128),128,
                       getDD(device),grid.getDD(device));
          __rtc_launch(rtc,umeshClampRangeLow,
                       divRoundUp(dims.x*dims.y*dims.z,1024),1024,
                       grid.getDD(device),dims.x*dims.y*dims.z);
        }
        for (auto device : *devices)
          device->sync();
        return;
      }

      for (auto device : *devices) 
        __rtc_launch(device->rtc,
                     umeshRasterCells,
                     divRoundUp(numCells,128),128,
                     getDD(device),grid.getDD(device));
      for (auto device : *devices)
        device->sync();
    }
    
    bool UMeshField::set1i(const std::string &member, const int &value)
    {
      if (member == "numChannels") { numChannels = value; return true; }
      if (member == "channelMask") {
        /* a mask change rewrites the per-vertex active vectors, the macro-cell
           ranges and the majorants; see setChannelMask */
        setChannelMask((uint32_t)value);
        return true;
      }
      return ScalarField::set1i(member,value);
    }

    bool UMeshField::setData(const std::string &member,
                             const std::shared_ptr<Data> &value)
    {
      if (ScalarField::setData(member,value))
        return true;

      if (member == "channel.vector") {
        /* numChannels*numVertices vec3f, channel-major: the per-channel
           current vector at each vertex */
        channelVecData = value->as<PODData>();
        return true;
      }

      if (member == "cell.index") {
        cellOffsets = value->as<PODData>();
        return true;
      }
      if (member == "cell.type") {
        cellTypes = value->as<PODData>();
        return true;
      }
      if (member == "cell.data") {
        scalars = value->as<PODData>();
        scalarsArePerVertex = false;
        return true;
      }
      if (member == "vertex.position") {
        vertices = value->as<PODData>();
        return true;
      }
      if (member == "vertex.data") {
        scalars = value->as<PODData>();
        scalarsArePerVertex = true;
        return true;
      }
      if (member == "index") {
        indices = value->as<PODData>();
        return true;
      }

      return false;
    }

#if RTC_DEVICE_CODE
    /*! cuBQL BVH build (refit_init) assumes finite, non-inverted prim boxes.
      Invalid connectivity (e.g. OOB vertex indices) yields empty / ±inf
      bounds from cellBounds — sanitize to a tiny finite box so the builder
      does not corrupt GPU memory; sampling still rejects bad cells in
      eltScalar / tetScalar. */
    inline __rtc_device bool umeshFiniteF(float x)
    {
      return (x == x) && (fabsf(x) < 1e37f);
    }

    inline __rtc_device bool umeshPrimBoundsSafeForCuBQL(const box4f &eb)
    {
      const box3f pb = getBox(eb);
      const range1f rw = getRange(eb);
      if (pb.empty())
        return false;
      if (!umeshFiniteF(pb.lower.x) || !umeshFiniteF(pb.lower.y)
          || !umeshFiniteF(pb.lower.z) || !umeshFiniteF(pb.upper.x)
          || !umeshFiniteF(pb.upper.y) || !umeshFiniteF(pb.upper.z))
        return false;
      if (!umeshFiniteF(rw.lower) || !umeshFiniteF(rw.upper))
        return false;
      if (rw.lower > rw.upper)
        return false;
      return true;
    }

    inline __rtc_device box4f umeshSanitizePrimBoundsForCuBQL(const UMeshField::DD &mesh,
                                                              const box4f &eb)
    {
      if (umeshPrimBoundsSafeForCuBQL(eb))
        return eb;
      const box3f &wb = mesh.worldBounds;
      if (!wb.empty() && umeshFiniteF(wb.lower.x) && umeshFiniteF(wb.lower.y)
          && umeshFiniteF(wb.lower.z) && umeshFiniteF(wb.upper.x)
          && umeshFiniteF(wb.upper.y) && umeshFiniteF(wb.upper.z)) {
        const vec3f mid = 0.5f * (wb.lower + wb.upper);
        const float pad = 1e-3f;
        const vec3f lo(mid.x - pad, mid.y - pad, mid.z - pad);
        const vec3f hi(mid.x + pad, mid.y + pad, mid.z + pad);
        return box4f(vec4f(lo, 0.f), vec4f(hi, 1.f));
      }
      return box4f(vec4f(0.f, 0.f, 0.f, 0.f), vec4f(1.f, 1.f, 1.f, 1.f));
    }
#endif
    
    __rtc_global 
    void umeshComputeElementBBs(rtc::ComputeInterface ci,
                                UMeshField::DD mesh,
                                box3f   *d_primBounds,
                                range1f *d_primRanges)
    {
#if RTC_DEVICE_CODE
      const int tid = ci.launchIndex().x;
      if (tid >= mesh.numCells) return;
    
      const box4f eb = umeshSanitizePrimBoundsForCuBQL(mesh, mesh.cellBounds(tid));
      d_primBounds[tid] = getBox(eb);
      if (d_primRanges) d_primRanges[tid] = getRange(eb);
#endif
    }
  
    /*! computes, on specified device, the bounding boxes and - if
      d_primRanges is non-null - the primitmives ranges. d_primBounds
      and d_primRanges (if non-null) must be pre-allocated and
      writeaable on specified device */
    void UMeshField::computeElementBBs(Device  *device,
                                       box3f   *d_primBounds,
                                       range1f *d_primRanges)
    {
      __rtc_launch(device->rtc,umeshComputeElementBBs,
                   divRoundUp(numCells,128),128,
                   getDD(device),d_primBounds,d_primRanges);
      device->rtc->sync();
    }

  
    UMeshField::UMeshField(Context *context, 
                           const DevGroup::SP   &devices)
      : ScalarField(context,devices)
    {
      perLogical.resize(devices->numLogical);
    }

    __rtc_global
    void computeWorldBounds(rtc::ComputeInterface ci,
                            UMeshField::DD mesh,
                            box3f *pWorldBounds)
    {
#if RTC_DEVICE_CODE
      int tid = ci.launchIndex().x;
      if (tid >= mesh.numCells)
        return;
      box4f bounds = mesh.cellBounds(tid);
      rtc::fatomicMin(&pWorldBounds->lower.x,bounds.lower.x);
      rtc::fatomicMin(&pWorldBounds->lower.y,bounds.lower.y);
      rtc::fatomicMin(&pWorldBounds->lower.z,bounds.lower.z); 
      rtc::fatomicMax(&pWorldBounds->upper.x,bounds.upper.x);
      rtc::fatomicMax(&pWorldBounds->upper.y,bounds.upper.y);
      rtc::fatomicMax(&pWorldBounds->upper.z,bounds.upper.z);
#endif
    }
  
    void UMeshField::commit()
    {
      assert(indices);
      assert(vertices);
      assert(cellOffsets);
      assert(cellTypes);
      assert(scalars);
    
      this->numCells = (int)cellOffsets->count;
      for (auto device : *devices) {
        PLD *pld = getPLD(device); 
        auto rtc = device->rtc;
        SetActiveGPU forDuration(device);

        int numCells = (int)cellOffsets->count;
        assert(numCells > 0);
        int numIndices  = (int)indices->count;
        if (!pld->pWorldBounds) {
          pld->pWorldBounds
            = (box3f*)rtc->allocMem(sizeof(box3f));
        }
        box3f emptyBox;
        rtc->copy(pld->pWorldBounds,&emptyBox,sizeof(emptyBox));
        __rtc_launch(rtc,computeWorldBounds,
                     divRoundUp(numCells,128),128,
                     getDD(device),pld->pWorldBounds);
      }
    
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        device->sync();
        // in case of having multiple devices this will repeately
        // download the same value; that's ok.
        PLD *pld = getPLD(device); 
        device->rtc->copy(&worldBounds,
                          pld->pWorldBounds,
                          sizeof(worldBounds));
        device->sync();
        assert(!worldBounds.empty());
      }
    }
  

    UMeshField::DD UMeshField::getDD(Device *device)
    {
      assert(device);
      UMeshField::DD dd;
      assert(vertices);
      assert(indices);
      (ScalarField::DD &)dd = ScalarField::getDD(device);
      dd.vertices    = (const vec3f *)vertices->getDD(device);
      dd.scalars     = (const float *)scalars->getDD(device);
      dd.indices     = (const int   *)indices->getDD(device);
      dd.cellOffsets = (const int   *)cellOffsets->getDD(device);
      dd.cellTypes   = (const uint8_t *)cellTypes->getDD(device);
      dd.scalarsArePerVertex = scalarsArePerVertex;
      dd.numCells    = (int)cellOffsets->count;
      dd.numVertices = (int)vertices->count;
      dd.numScalars  = (int)scalars->count;
      dd.numIndices  = (int)indices->count;
      dd.numChannels = numChannels;
      dd.channelMask = channelMask;
      dd.channelVec  = channelVecData
        ? (const vec3f *)channelVecData->getDD(device) : nullptr;
      dd.activeVec   = (const vec3f *)getPLD(device)->activeVec;
      assert(dd.numCells > 0);
      assert(dd.vertices);
      assert(dd.indices);
      return dd;
    }
  
    IsoSurfaceAccel::SP UMeshField::createIsoAccel(IsoSurface *isoSurface) 
    {
      auto sampler = std::make_shared<UMeshCuBQLSampler>(this);
      return std::make_shared<MCIsoSurfaceAccel<UMeshCuBQLSampler>>
        (isoSurface,
         createGeomType_UMeshMC_Iso,
         sampler);
    }
  
    VolumeAccel::SP UMeshField::createAccel(Volume *volume)
    {
#if 0
      return std::make_shared<AWTAccel>(volume,this);
#else
      auto sampler
        = std::make_shared<UMeshCuBQLSampler>(this);
      auto accel = std::make_shared<MCVolumeAccel<UMeshCuBQLSampler>>
        (volume,
         createGeomType_UMeshMC,
         sampler);
      /* a channel-mask change commits the field, not the volume, so the field
         has to be able to refresh this accel's majorants itself */
      accels.push_back(accel);
      return accel;
#endif
    }
  }
}

