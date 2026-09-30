// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/volume/ParticleRBFSampler.h"

namespace BARNEY_NS {
  namespace native {

    __rtc_global
    void RBF_initProbes(const rtc::ComputeInterface &ci,
                        float *cellMin, float *cellMax, int numCells)
    {
#if RTC_DEVICE_CODE
      const int i = ci.launchIndex().x;
      if (i >= numCells) return;
      /* inverted, so a cell no blob centre lands in is recognisable */
      cellMin[i] = +1e30f;
      cellMax[i] = -1e30f;
#endif
    }

    /*! Sample the field once at every macro-cell *corner* that borders an
        occupied cell.

        With samplesPerAxis==2 the lattice probes are exactly the 8 corners of a
        cell, and every interior corner is shared by 8 cells - so probing per
        cell evaluates each corner 8 times over. At mcGridSize=1024 that is 20M
        field evaluations where 2.6M distinct ones exist. The values are
        identical either way; this just stops computing them repeatedly. */
    __rtc_global
    void RBF_probeCorners(const rtc::ComputeInterface &ci,
                          ParticleRBFSampler::DD sampler,
                          MCGrid::DD grid,
                          float *corners)
    {
#if RTC_DEVICE_CODE
      const int idx = ci.launchIndex().x;
      const vec3i cdims = grid.dims + vec3i(1);
      const int nc = cdims.x*cdims.y*cdims.z;
      if (idx >= nc) return;
      const vec3i c = vec3i(idx % cdims.x,
                            (idx / cdims.x) % cdims.y,
                            idx / (cdims.x*cdims.y));

      /* only corners touching an occupied cell are worth sampling; a cell no
         primitive reaches is exactly zero and keeps that range */
      bool needed = false;
      for (int dz=-1;dz<=0 && !needed;dz++)
        for (int dy=-1;dy<=0 && !needed;dy++)
          for (int dx=-1;dx<=0 && !needed;dx++) {
            const vec3i n = c + vec3i(dx,dy,dz);
            if (n.x<0||n.y<0||n.z<0||
                n.x>=grid.dims.x||n.y>=grid.dims.y||n.z>=grid.dims.z) continue;
            const range1f r
              = grid.scalarRanges[n.x+grid.dims.x*(n.y+grid.dims.y*n.z)];
            if (!(r.lower==0.f && r.upper==0.f)) needed = true;
          }
      if (!needed) { corners[idx] = 0.f; return; }
      corners[idx] = sampler.sample(grid.gridOrigin + vec3f(c)*grid.gridSpacing);
#endif
    }

    __rtc_global
    void RBF_sampleMCRanges(const rtc::ComputeInterface &ci,
                            ParticleRBFSampler::DD sampler,
                            MCGrid::DD grid,
                            int numCells,
                            int samplesPerAxis,
                            float pad,
                            const float *probeMin,
                            const float *probeMax,
                            const float *corners)
    {
#if RTC_DEVICE_CODE
      const int cellID = ci.launchIndex().x;
      if (cellID >= numCells) return;

      /* scalarRanges still holds the conservative per-primitive bound here -
         this kernel is the only thing that narrows it, and it runs once */
      const range1f conservative = grid.scalarRanges[cellID];
      /* a cell no active primitive reaches is exactly zero, not merely bounded
         by zero, so leave it alone - that part of the bound is not an estimate */
      if (conservative.lower == 0.f && conservative.upper == 0.f) return;

      /* start from the blob centres scattered into this cell; they are where
         the field peaks, so they are the part of the estimate that matters */
      float mn = probeMin[cellID], mx = probeMax[cellID];

      if (samplesPerAxis == 1) {
        /* one probe at the cell centre. Worth having as its own case: the blob
           centres already cover where the field peaks, so the lattice is only
           there to say something about cells they missed, and 8 probes to say
           that costs 8x what 1 does. */
        const vec3i c = grid.cellID(cellID);
        const vec3f p = grid.gridOrigin
          + (vec3f(c)+vec3f(0.5f))*grid.gridSpacing;
        const float v = sampler.sample(p);
        mn = min(mn,v);
        mx = max(mx,v);
      } else if (samplesPerAxis == 2 && corners) {
        /* the 8 corners, read from the shared lattice rather than re-evaluated */
        const vec3i c = grid.cellID(cellID);
        const vec3i cdims = grid.dims + vec3i(1);
        for (int iz=0;iz<2;iz++)
          for (int iy=0;iy<2;iy++)
            for (int ix=0;ix<2;ix++) {
              const vec3i k = c + vec3i(ix,iy,iz);
              const float v = corners[k.x + cdims.x*(k.y + cdims.y*k.z)];
              mn = min(mn,v);
              mx = max(mx,v);
            }
      } else if (samplesPerAxis > 1) {
        const vec3i c = grid.cellID(cellID);
        const vec3f lo = grid.gridOrigin + vec3f(c)*grid.gridSpacing;
        const int N = samplesPerAxis;
        const float rcpN = 1.f/float(N-1);
        for (int iz=0;iz<N;iz++)
          for (int iy=0;iy<N;iy++)
            for (int ix=0;ix<N;ix++) {
              const vec3f f = vec3f(ix,iy,iz)*rcpN;
              const float v = sampler.sample(lo + f*grid.gridSpacing);
              mn = min(mn,v);
              mx = max(mx,v);
            }
      }

      /* nothing sampled this cell - no blob centre in it and no lattice - so
         the conservative bound is all we know, and it stays */
      if (mn > mx) return;

      const float span = mx-mn;
      mn -= pad*span;
      mx += pad*span;
      grid.scalarRanges[cellID]
        = range1f{ max(mn,conservative.lower), min(mx,conservative.upper) };
#endif
    }

    /*! Probe each blob at its own centre and widen the containing cell's range.

        A regular lattice of probes aliases badly against a field built from
        gaussians narrower than the probe spacing: the blobs here are 0.1-0.6 A
        wide, so a 3^3 lattice routinely misses every peak. The measured max then
        *under*-states the true one, and since that range becomes the woodcock
        majorant, the volume renders too transparent - measured at -17% to -56%
        of the field's energy, with the shortfall depending on which orbital
        channels are enabled, so it also broke comparisons between them.

        A blob peaks at its own centre, so probing there samples the extrema by
        construction. Driving this per *blob* rather than per cell is what keeps
        it affordable: one sample each, instead of a range query in every one of
        the ~10^6 cells. Cells that contain no blob centre keep the conservative
        bound, which is loose but never wrong. */
    __rtc_global
    void RBF_probeBlobCentres(const rtc::ComputeInterface &ci,
                              ParticleRBFSampler::DD sampler,
                              MCGrid::DD grid,
                              float *cellMin,
                              float *cellMax)
    {
#if RTC_DEVICE_CODE
      const int i = ci.launchIndex().x;
      if (i >= sampler.numActive) return;
      const int pid = (int)sampler.activeIDs[i];

      const vec3f p = sampler.centers[pid];
      vec3i c = vec3i((p-grid.gridOrigin)*rcp(grid.gridSpacing));
      c = min(max(c,vec3i(0)),grid.dims-vec3i(1));
      const size_t cellID
        = c.x + c.y*(size_t)grid.dims.x + c.z*(size_t)grid.dims.x*grid.dims.y;

      const float v = sampler.sample(p);
      rtc::fatomicMin(&cellMin[cellID],v);
      rtc::fatomicMax(&cellMax[cellID],v);
#endif
    }

    void ParticleRBFSampler::refineMCRanges(MCGrid::SP mcGrid,
                                            int samplesPerAxis,
                                            float pad)
    {
      if (!mcGrid) return;
      const int numCells = mcGrid->dims.x*mcGrid->dims.y*mcGrid->dims.z;
      for (auto device : *devices) {
        if (getPLD(device)->bvh.nodes == nullptr) continue;
        SetActiveGPU forDuration(device);
        auto rtc = device->rtc;

        /* negative disables refinement entirely and leaves the conservative
           per-primitive bound in place: exact, and the reference the blob-centre
           probing is measured against */
        if (samplesPerAxis < 0) continue;

        DD dd = getDD(device);
        if (dd.numActive == 0) continue;

        float *probeMin = (float *)rtc->allocMem(numCells*sizeof(float));
        float *probeMax = (float *)rtc->allocMem(numCells*sizeof(float));
        __rtc_launch(rtc,RBF_initProbes,
                     divRoundUp(numCells,1024),1024,
                     probeMin,probeMax,numCells);
        /* one sample per blob, scattered into the cell its centre falls in -
           see RBF_probeBlobCentres for why the peaks have to be sampled this
           way, and why it is driven per blob rather than per cell */
        __rtc_launch(rtc,RBF_probeBlobCentres,
                     divRoundUp(dd.numActive,128),128,
                     dd,mcGrid->getDD(device),probeMin,probeMax);
        /* the shared corner lattice; only for the default 2-probe case, where
           the lattice probes are exactly the cell corners */
        float *corners = nullptr;
        if (samplesPerAxis == 2) {
          const vec3i cd = mcGrid->dims + vec3i(1);
          const size_t nCorners = (size_t)cd.x*cd.y*cd.z;
          corners = (float *)rtc->allocMem(nCorners*sizeof(float));
          __rtc_launch(rtc,RBF_probeCorners,
                       divRoundUp((int)nCorners,128),128,
                       dd,mcGrid->getDD(device),corners);
        }
        __rtc_launch(rtc,RBF_sampleMCRanges,
                     divRoundUp(numCells,128),128,
                     dd,mcGrid->getDD(device),
                     numCells,samplesPerAxis,pad,probeMin,probeMax,corners);
        rtc->sync();
        if (corners) rtc->freeMem(corners);
        rtc->freeMem(probeMin);
        rtc->freeMem(probeMax);
      }
    }

    ParticleRBFSampler::ParticleRBFSampler(ParticleRBFField *field)
      : field(field),
        devices(field->devices)
    {
      perLogical.resize(devices->numLogical);
    }

    ParticleRBFSampler::PLD *
    ParticleRBFSampler::getPLD(Device *device)
    {
      assert(device);
      assert(device->contextRank() >= 0);
      assert(device->contextRank() < perLogical.size());
      return &perLogical[device->contextRank()];
    }

    ParticleRBFSampler::DD
    ParticleRBFSampler::getDD(Device *device)
    {
      DD dd;
      (ParticleRBFField::DD &)dd = field->getDD(device);
      dd.bvh = getPLD(device)->bvh;
      return dd;
    }

    /*! primitives per bvh leaf. cuBQL defaults this to 1, which for a field of
        400k blobs means 400k single-primitive leaves and a correspondingly deep
        tree. A range query here already has to visit every blob covering the
        point - 132 of them on average - so grouping them into leaves trades
        tree descent for a linear scan the query was going to do anyway.

        This changes only the order contributions are summed, not which ones, so
        the field is the same to within float associativity. */
    static int bvhLeafSize()
    {
      static int value = []{
        const char *s = getenv("BN_RBF_LEAF_SIZE");
        return s ? std::max(1,atoi(s)) : 8;
      }();
      return value;
    }

    void ParticleRBFSampler::build()
    {
      if (field->numParticles == 0)
        throw std::runtime_error("#bn.rbf: no primitives to build a bvh over");

      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        bvh_t &bvh = pld->bvh;
        const int numPrims = field->getPLD(device)->numActive;
        if (bvh.nodes != nullptr && pld->builtEpoch == field->filterEpoch)
          /* the active list has not been recomputed since this bvh was built */
          continue;
        if (bvh.nodes != nullptr) {
          SetActiveGPU forDuration(device);
#if BARNEY_RTC_CPU || defined(__HIPCC__)
          cuBQL::cpu::freeBVH(bvh);
#else
          cuBQL::DeviceMemoryResource memResource;
          cuBQL::free(bvh,0,memResource);
#endif
          bvh = { 0,0 };
        }
        pld->builtEpoch = field->filterEpoch;
        if (numPrims == 0)
          /* everything is filtered out; leave an empty bvh and let sample()
             return zero everywhere */
          continue;

        SetActiveGPU forDuration(device);

        box3f *primBounds
          = (box3f*)device->rtc->allocMem(numPrims*sizeof(box3f));
        field->computeElementBBs(device,primBounds);
        device->rtc->sync();
#if BARNEY_RTC_CPU || defined(__HIPCC__)
        cuBQL::cpu::spatialMedian(bvh,
                                  (const cuBQL::box_t<float,3>*)primBounds,
                                  numPrims,
                                  cuBQL::BuildConfig(bvhLeafSize()));
#else
        /*! make sure to have cubql use regular device memory, not async
          mallocs; else we may allocate all memory on the first gpu */
        cuBQL::DeviceMemoryResource memResource;
        cuBQL::gpuBuilder(bvh,
                          (const cuBQL::box_t<float,3>*)primBounds,
                          numPrims,
                          cuBQL::BuildConfig(bvhLeafSize()),
                          0,
                          memResource);
#endif
        device->rtc->sync();
        device->rtc->freeMem(primBounds);

        std::cout << OWL_TERMINAL_LIGHT_GREEN
                  << "#bn.rbf: cubql bvh built over "
                  << numPrims << " of " << field->numParticles << " primitives"
                  << OWL_TERMINAL_DEFAULT << std::endl;
      }
    }

  }
}
