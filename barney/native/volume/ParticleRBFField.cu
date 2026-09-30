// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/volume/ParticleRBFField.h"
#include "native/Context.h"
#include "native/volume/MCGrid.cuh"
#include "native/volume/ParticleRBFSampler.h"
#include <chrono>

namespace BARNEY_NS {
  namespace native {

    RTC_IMPORT_USER_GEOM(/*file*/ParticleRBFMC,
                         /*name*/ParticleRBFMC,
                         /*geomtype device data */
                         MCVolumeAccel<ParticleRBFSampler>::DD,false,false);

    // ==================================================================
    // world bounds
    // ==================================================================

    __rtc_global
    void RBF_computeWorldBounds(const rtc::ComputeInterface &ci,
                                box3f *pBounds,
                                const ParticleRBFField::DD dd)
    {
#if RTC_DEVICE_CODE
      const int tid = ci.launchIndex().x;
      if (tid >= dd.numParticles) return;

      const box3f bb = dd.particleBounds(tid);
      rtc::fatomicMin(&pBounds->lower.x,bb.lower.x);
      rtc::fatomicMin(&pBounds->lower.y,bb.lower.y);
      rtc::fatomicMin(&pBounds->lower.z,bb.lower.z);
      rtc::fatomicMax(&pBounds->upper.x,bb.upper.x);
      rtc::fatomicMax(&pBounds->upper.y,bb.upper.y);
      rtc::fatomicMax(&pBounds->upper.z,bb.upper.z);
#endif
    }

    // ==================================================================
    // the active list: which primitives the filter keeps
    // ==================================================================

    /*! membership only, in primitive order, so it can be compared run to run -
        which the compacted list below cannot, because it comes out in atomic
        order and is a different permutation every time it is rebuilt */
    __rtc_global
    void RBF_markActive(const rtc::ComputeInterface &ci,
                        const ParticleRBFField::DD dd,
                        uint8_t *kept)
    {
#if RTC_DEVICE_CODE
      const int tid = ci.launchIndex().x;
      if (tid >= dd.numParticles) return;
      kept[tid] = (dd.groupIDs && !dd.filter.passes(dd.groupIDs[tid])) ? 0 : 1;
#endif
    }

    __rtc_global
    void RBF_compactActive(const rtc::ComputeInterface &ci,
                           const ParticleRBFField::DD dd,
                           uint32_t *activeIDs,
                           int *counter)
    {
#if RTC_DEVICE_CODE
      const int tid = ci.launchIndex().x;
      if (tid >= dd.numParticles) return;
      if (dd.groupIDs && !dd.filter.passes(dd.groupIDs[tid])) return;
      /* order does not matter to a bvh build, so an atomic counter is enough
         and saves a prefix sum over a couple of million primitives */
      activeIDs[ci.atomicAdd(counter,1)] = (uint32_t)tid;
#endif
    }

    // ==================================================================
    // channel recontraction: sum the enabled channels into one dense
    // polynomial per blob
    // ==================================================================

    __rtc_global
    void RBF_recontract(const rtc::ComputeInterface &ci,
                        const ParticleRBFField::DD dd,
                        float *dense)
    {
#if RTC_DEVICE_CODE
      const int pid = ci.launchIndex().x;
      if (pid >= dd.numParticles) return;

      float *out = dense + (size_t)pid*dd.polyStride;
      for (int i=0;i<dd.polyStride;i++) out[i] = 0.f;

      const int base = pid*dd.numChannels;
      for (int c=0;c<dd.numChannels;c++) {
        if (!((dd.channelMask >> c) & 1u)) continue;
        const uint32_t hi = dd.polyOffsets[base+c+1];
        for (uint32_t t=dd.polyOffsets[base+c];t<hi;t++) {
          const uint32_t m = dd.polyMonomials[t];
          /* one thread owns this blob, so the accumulation needs no atomics */
          out[rbf_monomialSlot((m)&0xff,(m>>8)&0xff,(m>>16)&0xff,
                               dd.polyMaxDegree)]
            += dd.polyCoeffs[t];
        }
      }
#endif
    }

    // ==================================================================
    // per-primitive bounds, for the cuBQL build
    // ==================================================================

    __rtc_global
    void RBF_computeElementBBs(const rtc::ComputeInterface &ci,
                               box3f *d_primBounds,
                               const ParticleRBFField::DD dd)
    {
#if RTC_DEVICE_CODE
      const int tid = ci.launchIndex().x;
      if (tid >= dd.numActive) return;
      d_primBounds[tid] = dd.particleBounds(dd.activeIDs[tid]);
#endif
    }

    // ==================================================================
    // macro cells
    // ==================================================================

    __rtc_global
    void RBF_clearBounds(const rtc::ComputeInterface &ci,
                         float *posBound,
                         float *negBound,
                         int numCells)
    {
#if RTC_DEVICE_CODE
      const int tid = ci.launchIndex().x;
      if (tid >= numCells) return;
      posBound[tid] = 0.f;
      negBound[tid] = 0.f;
#endif
    }

    /*! accumulates each primitive's largest possible |contribution| into every
        macro-cell its support touches. the field is a *signed sum*, so unlike
        the interpolating field types a per-cell min/max over primitives would
        not be conservative - contributions add, so the bounds have to add
        too. the per-cell distance interval keeps that from being as loose as a
        global per-primitive peak would be. */
    __rtc_global
    void RBF_rasterParticles(const rtc::ComputeInterface &ci,
                             const ParticleRBFField::DD dd,
                             MCGrid::DD grid,
                             float *posBound,
                             float *negBound)
    {
#if RTC_DEVICE_CODE
      const int i = ci.launchIndex().x;
      if (i >= dd.numActive) return;
      const int tid = (int)dd.activeIDs[i];

      const vec3f center = dd.centers[tid];
      const box3f pb = dd.particleBounds(tid);

      vec3i lo = vec3i((pb.lower-grid.gridOrigin)*rcp(grid.gridSpacing));
      vec3i hi = vec3i((pb.upper-grid.gridOrigin)*rcp(grid.gridSpacing));
      lo = min(max(lo,vec3i(0)),grid.dims-vec3i(1));
      hi = min(max(hi,vec3i(0)),grid.dims-vec3i(1));

      const bool oneSided = dd.signDefinite(tid);
      const bool positive = dd.coeffs[tid] > 0.f;

      for (int iz=lo.z;iz<=hi.z;iz++)
        for (int iy=lo.y;iy<=hi.y;iy++)
          for (int ix=lo.x;ix<=hi.x;ix++) {
            const vec3f cellLo
              = grid.gridOrigin + vec3f(ix,iy,iz)*grid.gridSpacing;
            const vec3f cellHi = cellLo + grid.gridSpacing;

            /* nearest and farthest point of the cell from the center, so the
               bound only has to hold over the distances actually reachable */
            const vec3f outside
              = max(vec3f(0.f),max(cellLo-center,center-cellHi));
            const vec3f corner
              = max(abs(cellLo-center),abs(cellHi-center));
            const float rLo = length(outside);
            const float rHi = length(corner);

            const float b = dd.particleBound(tid,rLo,rHi);
            if (b == 0.f) continue;

            const size_t cellID
              = ix
              + iy * (size_t)grid.dims.x
              + iz * (size_t)grid.dims.x * (size_t)grid.dims.y;
            if (oneSided) {
              if (positive) ci.atomicAdd(&posBound[cellID],b);
              else          ci.atomicAdd(&negBound[cellID],b);
            } else {
              ci.atomicAdd(&posBound[cellID],b);
              ci.atomicAdd(&negBound[cellID],b);
            }
          }
#endif
    }

    __rtc_global
    void RBF_finishBounds(const rtc::ComputeInterface &ci,
                          MCGrid::DD grid,
                          const float *posBound,
                          const float *negBound,
                          int numCells)
    {
#if RTC_DEVICE_CODE
      const int tid = ci.launchIndex().x;
      if (tid >= numCells) return;
      grid.scalarRanges[tid] = range1f{ -negBound[tid], posBound[tid] };
#endif
    }

    // ==================================================================

    ParticleRBFField::ParticleRBFField(Context *context,
                                       const DevGroup::SP &devices)
      : ScalarField(context,devices)
    {
      perLogical.resize(devices->numLogical);
    }

    ParticleRBFField::PLD *ParticleRBFField::getPLD(Device *device)
    {
      assert(device);
      assert(device->contextRank() >= 0);
      assert(device->contextRank() < perLogical.size());
      return &perLogical[device->contextRank()];
    }

    void ParticleRBFField::updateActiveList()
    {
      bool anyChanged = false;
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto rtc = device->rtc;
        PLD *pld = getPLD(device);

        if (pld->capacity < (size_t)numParticles) {
          if (pld->activeIDs) rtc->freeMem(pld->activeIDs);
          if (pld->kept)      rtc->freeMem(pld->kept);
          pld->activeIDs
            = (uint32_t *)rtc->allocMem(numParticles*sizeof(uint32_t));
          pld->kept = (uint8_t *)rtc->allocMem(numParticles*sizeof(uint8_t));
          pld->capacity = numParticles;
          pld->prevKept.clear();
        }

        /* An orbital-channel toggle changes no primitive's membership - every
           channel lives on the same blob - so the active list, the bvh and the
           majorants all stay valid and none of them must be rebuilt for it.

           Deciding that has to come BEFORE recompacting: the compaction writes
           the list in atomic order, so re-running it permutes the list, and the
           bvh indexes *into* that list. Rebuilding one without the other points
           every bvh leaf at a different primitive - which renders as blobs
           scattered and clipped, and as a different image every time the same
           filter is toggled.

           Comparing membership flags rather than the count is also deliberate:
           two different selections very easily contain the same number of
           primitives here, because symmetry-equivalent bonds come in
           equal-sized families. */
        __rtc_launch(rtc,RBF_markActive,
                     divRoundUp(numParticles,128),128,
                     getDD(device),pld->kept);
        rtc->sync();
        std::vector<uint8_t> nowKept(numParticles);
        rtc->copy(nowKept.data(),pld->kept,numParticles*sizeof(uint8_t));
        if (pld->prevKept == nowKept)
          continue;
        pld->prevKept.swap(nowKept);
        anyChanged = true;

        int zero = 0;
        int *d_counter = (int *)rtc->allocMem(sizeof(int));
        rtc->copy(d_counter,&zero,sizeof(zero));
        /* getDD() reads pld->numActive, but this kernel only writes the list,
           so the stale count it sees does not matter here */
        __rtc_launch(rtc,RBF_compactActive,
                     divRoundUp(numParticles,128),128,
                     getDD(device),pld->activeIDs,d_counter);
        rtc->sync();
        int n = 0;
        rtc->copy(&n,d_counter,sizeof(n));
        rtc->freeMem(d_counter);
        pld->numActive = n;
      }
      if (anyChanged) ++filterEpoch;
    }

    void ParticleRBFField::recontractChannels()
    {
      if (!particle.polyOffsets || numParticles == 0) return;
      const int stride = rbf_numCoeffs(polyMaxDegree);
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        /* the dense array depends on nothing but the mask, so a redundant
           commit - and every ui interaction causes one - must not redo it */
        if (pld->denseValid && pld->denseMask == channelMask) continue;

        SetActiveGPU forDuration(device);
        auto rtc = device->rtc;
        const size_t need = (size_t)numParticles*stride;
        if (pld->denseCapacity < need) {
          if (pld->polyDense) rtc->freeMem(pld->polyDense);
          pld->polyDense = (float *)rtc->allocMem(need*sizeof(float));
          pld->denseCapacity = need;
        }
        __rtc_launch(rtc,RBF_recontract,
                     divRoundUp(numParticles,128),128,
                     getDD(device),pld->polyDense);
        rtc->sync();
        pld->denseMask  = channelMask;
        pld->denseValid = true;
      }
    }

    ParticleRBFField::~ParticleRBFField()
    {}

    std::string ParticleRBFField::toString() const
    { return "barney::native::ParticleRBFField"; }

    ParticleRBFField::DD ParticleRBFField::getDD(Device *device)
    {
      ParticleRBFField::DD dd;

      // inherited:
      (ScalarField::DD &)dd = ScalarField::getDD(device);

      dd.centers      = (const vec3f *)particle.centers->getDD(device);
      dd.coeffs       = (const float *)particle.coeffs->getDD(device);
      dd.gammas       = (const float *)particle.gammas->getDD(device);
      dd.cutoffs      = (const float *)particle.cutoffs->getDD(device);
      dd.monomials    = particle.monomials
        ? (const uint32_t *)particle.monomials->getDD(device) : nullptr;
      dd.groupIDs     = particle.groupIDs
        ? (const uint32_t *)particle.groupIDs->getDD(device) : nullptr;
      dd.numParticles = numParticles;

      dd.polyOffsets   = particle.polyOffsets
        ? (const uint32_t *)particle.polyOffsets->getDD(device) : nullptr;
      dd.polyCoeffs    = particle.polyCoeffs
        ? (const float *)particle.polyCoeffs->getDD(device) : nullptr;
      dd.polyMonomials = particle.polyMonomials
        ? (const uint32_t *)particle.polyMonomials->getDD(device) : nullptr;
      dd.polyDense     = getPLD(device)->polyDense;
      dd.polyStride    = rbf_numCoeffs(polyMaxDegree);
      dd.directions    = group.directions
        ? (const vec3f *)group.directions->getDD(device) : nullptr;
      dd.numChannels   = numChannels;
      dd.channelMask   = channelMask;
      dd.vecMode       = vecMode;
      dd.polyMaxDegree = polyMaxDegree;

      PLD *pld = getPLD(device);
      dd.activeIDs    = pld->activeIDs;
      dd.numActive    = pld->numActive;

      dd.filter.groupEnabled = group.enabled
        ? (const uint8_t *)group.enabled->getDD(device) : nullptr;
      dd.filter.groupValues  = group.values
        ? (const float *)group.values->getDD(device) : nullptr;
      dd.filter.valueRange   = filterValueRange;
      dd.filter.numGroups    = numGroups;

      return dd;
    }

    void ParticleRBFField::computeElementBBs(Device *device,
                                             box3f *d_primBounds)
    {
      const int bs = 128;
      const int nb = divRoundUp(max(1,getPLD(device)->numActive),bs);
      __rtc_launch(device->rtc, RBF_computeElementBBs,
                   nb,bs,
                   d_primBounds,
                   getDD(device));
      device->sync();
    }

    MCGrid::SP ParticleRBFField::buildMCs()
    {
      if (mcGrid) return mcGrid;
      if (getPLD((*devices)[0])->numActive == 0)
        updateActiveList();

      mcGrid = std::make_shared<MCGrid>(devices);
      auto &grid = *mcGrid;

      const float maxWidth = reduce_max(getBox(worldBounds).size());
      vec3i dims
        = 1+vec3i(getBox(worldBounds).size() * ((mcGridSize-1) / maxWidth));
      dims = max(dims,vec3i(1));

      std::cout << OWL_TERMINAL_BLUE
                << "#bn.rbf: building macro cell grid of "
                << dims.x << "x" << dims.y << "x" << dims.z
                << OWL_TERMINAL_DEFAULT << std::endl;

      grid.resize(dims);
      grid.gridOrigin  = worldBounds.lower;
      grid.gridSpacing = worldBounds.size() * rcp(vec3f(dims));

      /* the refinement inside refreshMCs() samples the field, so the bvh has
         to exist first; the accel builds it after us otherwise */
      if (sampler) sampler->build();
      refreshMCs();
      return mcGrid;
    }

    void ParticleRBFField::refreshMCs()
    {
      if (!mcGrid) return;
      const auto tMC = std::chrono::steady_clock::now();
      auto &grid = *mcGrid;
      const int numCells = grid.dims.x*grid.dims.y*grid.dims.z;

      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto rtc = device->rtc;

        float *posBound = (float *)rtc->allocMem(numCells*sizeof(float));
        float *negBound = (float *)rtc->allocMem(numCells*sizeof(float));

        __rtc_launch(rtc,RBF_clearBounds,
                     divRoundUp(numCells,1024),1024,
                     posBound,negBound,numCells);
        __rtc_launch(rtc,RBF_rasterParticles,
                     divRoundUp(max(1,getPLD(device)->numActive),128),128,
                     getDD(device),grid.getDD(device),
                     posBound,negBound);
        __rtc_launch(rtc,RBF_finishBounds,
                     divRoundUp(numCells,1024),1024,
                     grid.getDD(device),posBound,negBound,numCells);
        rtc->sync();

        rtc->freeMem(posBound);
        rtc->freeMem(negBound);
      }
      const double tRaster = std::chrono::duration<double,std::milli>
        (std::chrono::steady_clock::now()-tMC).count();

      /* 0 is meaningful now: it means "probe blob centres only", which is both
         the cheapest and the most accurate estimator for this field. Only a
         negative value skips refinement altogether. */
      if (sampler && mcSamplesPerAxis >= 0)
        sampler->refineMCRanges(mcGrid,mcSamplesPerAxis,mcRangePad);
      /* Majorants are derived from the ranges we just rewrote, and a filter
         change commits the field rather than the volume, so Volume::commit()
         never runs and nothing else would refresh them. A majorant left over
         from the previous ranges is too small wherever the new ones are
         larger, and woodcock tracking then misses extinction - which renders
         as blobs clipped along macro-cell faces, and as a different image
         every time the same filter is toggled.

         rebuildMajorantsOnly() is a no-op until the accel has built once, so
         this is safe on the call that comes from buildMCs(). */
      ++mcGrid->contentEpoch;
      for (auto &weak : accels)
        if (auto accel = weak.lock())
          accel->rebuildMajorantsOnly();
      const double tAll = std::chrono::duration<double,std::milli>
        (std::chrono::steady_clock::now()-tMC).count();
      if (getenv("BN_RBF_DEBUG_MC")) {
        /* checksum the ranges so a second refresh can be compared with the
           first: if these match, anything that still differs is downstream */
        const int nc = mcGrid->dims.x*mcGrid->dims.y*mcGrid->dims.z;
        std::vector<range1f> h(nc);
        auto dev = (*devices)[0];
        SetActiveGPU forDuration(dev);
        dev->rtc->copy(h.data(),mcGrid->getDD(dev).scalarRanges,
                       nc*sizeof(range1f));
        double lo=0,hi=0; int nonEmpty=0;
        for (auto &r : h) {
          lo += r.lower; hi += r.upper;
          if (!(r.lower==0.f && r.upper==0.f)) nonEmpty++;
        }
        std::cout << "#bn.rbf[mc] cells=" << nc << " nonEmpty=" << nonEmpty
                  << " sumLo=" << lo << " sumHi=" << hi
                  << " mask=0x" << std::hex << channelMask << std::dec
                  << " numActive=" << getPLD(dev)->numActive
                  << " epoch=" << filterEpoch << std::endl;
      }
      /* this is the latency a channel toggle costs, so it is worth seeing
         rather than guessing at; quiet for the small ones */
      if (tAll > 200.0)
        std::cout << "#bn.rbf: macro-cell refresh " << tAll
                  << " ms (conservative raster " << tRaster
                  << ", refine " << (tAll-tRaster) << ")" << std::endl;
    }

    VolumeAccel::SP ParticleRBFField::createAccel(Volume *volume)
    {
      if (!sampler)
        sampler = std::make_shared<ParticleRBFSampler>(this);
      auto accel = std::make_shared<MCVolumeAccel<ParticleRBFSampler>>
        (volume,
         createGeomType_ParticleRBFMC,
         sampler);
      accels.push_back(accel);
      return accel;
    }

    bool ParticleRBFField::setData(const std::string &member,
                                   const std::shared_ptr<Data> &value)
    {
      if (ScalarField::setData(member,value)) return true;

      if (member == "particle.center")   { particle.centers   = value->as<PODData>(); return true; }
      if (member == "particle.coeff")    { particle.coeffs    = value->as<PODData>(); return true; }
      if (member == "particle.gamma")    { particle.gammas    = value->as<PODData>(); return true; }
      if (member == "particle.cutoff")   { particle.cutoffs   = value->as<PODData>(); return true; }
      if (member == "particle.monomial") { particle.monomials = value->as<PODData>(); return true; }
      if (member == "particle.group")    { particle.groupIDs  = value->as<PODData>(); return true; }
      if (member == "particle.polyOffset")   { particle.polyOffsets   = value->as<PODData>(); return true; }
      if (member == "particle.polyCoeff")    { particle.polyCoeffs    = value->as<PODData>(); return true; }
      if (member == "particle.polyMonomial") { particle.polyMonomials = value->as<PODData>(); return true; }
      if (member == "group.direction")       { group.directions       = value->as<PODData>(); return true; }
      if (member == "group.enabled")     { group.enabled      = value->as<PODData>(); return true; }
      if (member == "group.value")       { group.values       = value->as<PODData>(); return true; }

      return false;
    }

    bool ParticleRBFField::set1f(const std::string &member, const float &value)
    {
      if (member == "mcRangePad") { mcRangePad = max(0.f,value); return true; }
      if (member == "filter.valueMin") { filterValueRange.lower = value; return true; }
      if (member == "filter.valueMax") { filterValueRange.upper = value; return true; }
      return false;
    }

    bool ParticleRBFField::set2f(const std::string &member, const vec2f &value)
    {
      if (member == "filter.valueRange") {
        filterValueRange = range1f{value.x,value.y};
        return true;
      }
      return false;
    }

    bool ParticleRBFField::set3f(const std::string &member, const vec3f &value)
    {
      if (member == "bounds.lower") { explicitBounds.lower = value; return true; }
      if (member == "bounds.upper") { explicitBounds.upper = value; return true; }
      return false;
    }

    bool ParticleRBFField::set1i(const std::string &member, const int &value)
    {
      if (member == "mcGridSize") { mcGridSize = max(2,value); return true; }
      if (member == "mcSamplesPerAxis") { mcSamplesPerAxis = value; return true; }
      if (member == "numGroups")  { numGroups = value; return true; }
      if (member == "numChannels") { numChannels = max(1,value); return true; }
      /* every orbital channel lives on the same blob's centre, exponent and
         cutoff, so switching channels moves no geometry: updateActiveList()
         sees identical membership and leaves the bvh alone. the macro-cell
         *ranges* do get refreshed, because those are value ranges and the
         value did change. */
      if (member == "channelMask") { channelMask = (uint32_t)value; return true; }
      if (member == "vecMode")     { vecMode = value; return true; }
      if (member == "polyMaxDegree") {
        polyMaxDegree = max(0,min((int)RBF_MAX_DEGREE,value)); return true;
      }
      return false;
    }

    void ParticleRBFField::commit()
    {
      if (!particle.centers)
        throw std::runtime_error("#bn.rbf: missing 'particle.center'");
      if (!particle.coeffs)
        throw std::runtime_error("#bn.rbf: missing 'particle.coeff'");
      if (!particle.gammas)
        throw std::runtime_error("#bn.rbf: missing 'particle.gamma'");
      if (!particle.cutoffs)
        throw std::runtime_error("#bn.rbf: missing 'particle.cutoff'");

      numParticles = (int)particle.centers->count;
      assert(numParticles > 0);
      assert(particle.coeffs->count  == numParticles);
      assert(particle.gammas->count  == numParticles);
      assert(particle.cutoffs->count == numParticles);

      if (numGroups == 0) {
        if (group.values)       numGroups = (int)group.values->count;
        else if (group.enabled) numGroups = (int)group.enabled->count;
      }

      // ==================================================================
      // world bounds
      // ==================================================================
      if (!explicitBounds.empty()) {
        worldBounds = explicitBounds;
      } else {
        auto dev = (*devices)[0];
        SetActiveGPU forDuration(dev);
        auto rtc = dev->rtc;
        worldBounds = box3f();
        box3f *d_worldBounds = (box3f *)rtc->allocMem(sizeof(box3f));
        rtc->copy(d_worldBounds,&worldBounds,sizeof(worldBounds));
        __rtc_launch(rtc,RBF_computeWorldBounds,
                     divRoundUp(numParticles,128),128,
                     d_worldBounds,getDD(dev));
        rtc->sync();
        rtc->copy(&worldBounds,d_worldBounds,sizeof(worldBounds));
        rtc->freeMem(d_worldBounds);
      }

      /* the dense polynomial has to exist before anything samples the field,
         and has to be rebuilt before the majorants that are measured from it */
      const auto t0 = std::chrono::steady_clock::now();
      recontractChannels();

      /* the filter can change after the first build. both the bvh and the
         majorants are derived from it, so recompact first and then redo
         whichever of them already exists. */
      updateActiveList();
      if (sampler) sampler->build();
      if (mcGrid) refreshMCs();

      /* a filter change goes through here, so this is the latency the user
         feels when they tick a channel; worth seeing rather than guessing at */
      const double ms = std::chrono::duration<double,std::milli>
        (std::chrono::steady_clock::now()-t0).count();
      if (mcGrid && ms > 200.0)
        std::cout << "#bn.rbf: filter update took " << ms << " ms" << std::endl;
    }
  }
}
