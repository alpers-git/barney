// SPDX-FileCopyrightText:
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier:
// Apache-2.0


#include "barney/amr/BlockStructuredCuBQLSampler.h"

namespace BARNEY_NS {

  BlockStructuredCuBQLSampler
  ::BlockStructuredCuBQLSampler(BlockStructuredField *field)
    : field(field),
      devices(field->devices)
  {
    perLogical.resize(devices->numLogical);
  }

  BlockStructuredCuBQLSampler::PLD *
  BlockStructuredCuBQLSampler::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }
  
  BlockStructuredCuBQLSampler::DD
  BlockStructuredCuBQLSampler::getDD(Device *device)
  {
    DD dd;
    (BlockStructuredField::DD &)dd = field->getDD(device);
    dd.bvh    = getPLD(device)->bvh;
    dd.active = getPLD(device)->active;
    return dd;
  }

  /*! per-cell coverage kernel: a cell is *inactive* (0) when a finer block
      refines it, i.e. cellAt at its centre returns a smaller cell than its own.
      Those covered coarse cells are the ones Chombo keeps under refined regions;
      excluding them from the basis sum is what removes the boundary bands. */
  __rtc_global
  void BSSampler_computeActive(const rtc::ComputeInterface &ci,
                               BlockStructuredCuBQLSampler::DD samp,
                               uint8_t *active)
  {
#if RTC_DEVICE_CODE
    const int bid = ci.launchIndex().x;
    if (bid >= samp.numBlocks) return;
    const Block block = Block::getFrom(samp,bid);
    const uint64_t goff = samp.perBlock.offsets[bid];
    const float h = block.cellSize;
    for (int z=0;z<block.dims.z;z++)
      for (int y=0;y<block.dims.y;y++)
        for (int x=0;x<block.dims.x;x++) {
          const vec3f center
            = (vec3f(block.origin+vec3i(x,y,z))+vec3f(0.5f))*h;
          const AMRLeaf f = samp.cellAt(center);
          const bool covered = f.found && (f.cellSize < h*0.9999f);
          const uint64_t lidx
            = (uint64_t)x + block.dims.x*((uint64_t)y + block.dims.y*(uint64_t)z);
          active[goff+lidx] = covered ? 0 : 1;
        }
#endif
  }

  void BlockStructuredCuBQLSampler::build()
  {
    int numPrims = field->numBlocks;
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      bvh_t &bvh = pld->bvh;
      if (bvh.nodes != nullptr) {
        /* BVH already built! */
        continue;
      }

      std::cout << "------------------------------------------" << std::endl;
      std::cout << "building BlockStructuredCuBQL BVH!" << std::endl;
      std::cout << "------------------------------------------" << std::endl;

      SetActiveGPU forDuration(device);

      box3f *primBounds
        = (box3f*)device->rtc->allocMem(numPrims*sizeof(box3f));
      range1f *valueRanges
        = (range1f*)device->rtc->allocMem(numPrims*sizeof(range1f));
      field->computeElementBBs(device,primBounds,valueRanges);
      device->rtc->sync();
#if BARNEY_RTC_EMBREE || defined(__HIPCC__)
      cuBQL::cpu::spatialMedian(bvh,
                                (const cuBQL::box_t<float,3>*)primBounds,
                                numPrims,
                                cuBQL::BuildConfig());
#else
      /*! make sure to have cubql use regular device memory, not async
        mallocs; else we may allocate all memory on the first gpu */
      cuBQL::DeviceMemoryResource memResource;
      cuBQL::gpuBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)primBounds,
                        numPrims,
                        cuBQL::BuildConfig(),
                        0,
                        memResource);
#endif
      device->rtc->sync();
      device->rtc->freeMem(primBounds);
      device->rtc->freeMem(valueRanges);

      // ---- per-cell active (non-covered) mask for the basis reconstruction ----
      // one byte per scalar; the bvh (just built) is needed to detect coverage
      const size_t numCells = field->scalars->count;
      pld->active = (uint8_t*)device->rtc->allocMem(numCells*sizeof(uint8_t));
      {
        const int bs = 128;
        const int nb = int((field->numBlocks + bs - 1) / bs);
        __rtc_launch(device->rtc, BSSampler_computeActive,
                     nb, bs, getDD(device), pld->active);
        device->rtc->sync();
      }

      std::cout << OWL_TERMINAL_LIGHT_GREEN
                << "#bn.bsfield: cubql bvh built ..."
                << OWL_TERMINAL_DEFAULT << std::endl;
    }
  }
  
}
  
