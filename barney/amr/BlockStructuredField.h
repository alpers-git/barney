// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/Object.h"
#include "barney/ModelSlot.h"
// pulls in the complete IsoSurfaceAccel / IsoSurface definitions so
// createIsoAccel() below can name IsoSurfaceAccel::SP (matches the
// structured / umesh field headers)
#include "barney/volume/MCAccelerator.h"

namespace BARNEY_NS {

  struct Block;

  /*! accumulator for the analytic basis-function gradient (ExaBricks, Wald et
      al. 2020, eq. 3): the reconstruction's numerator/denominator plus their
      per-axis partial derivatives, all summed over the active cells in the same
      single bvh query the value reconstruction uses. */
  struct BasisGrad {
    float D=0.f, N=0.f;             //!< sum_C H_C , sum_C H_C v_C
    float Dx=0.f,Dy=0.f,Dz=0.f;     //!< sum_C dH_C/d{x,y,z}
    float Nx=0.f,Ny=0.f,Nz=0.f;     //!< sum_C dH_C/d{x,y,z} * v_C
  };

  struct BlockStructuredField : public ScalarField
  {
    typedef std::shared_ptr<BlockStructuredField> SP;

    struct PLD {
      // rtc::ComputeKernel1D *mcRasterBlocks = 0;
      // rtc::ComputeKernel1D *computeElementBBs = 0;
      Block *blocks;
      float *scalars;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
    
    struct DD : public ScalarField::DD {

#if RTC_DEVICE_CODE
      /* compute basis function contribution of given block at point P, and add
         that to 'sumWeightedValues' and 'sumWeights'. returns true if P is
         inside the block *filter domain*, false if outside (in which case the
         out params are not defined) */
      inline __rtc_device bool addBasisFunctions(float &sumWeightedValues,
                                             float &sumWeights,
                                             uint32_t bid,
                                             vec3f P) const;
      /*! like addBasisFunctions, but also accumulates the analytic gradient
          terms (per-axis partials of the tent basis) into 'acc' for a given
          block - used to shade iso surfaces without central differences */
      inline __rtc_device void addBasisFunctionsGrad(BasisGrad &acc,
                                             uint32_t bid,
                                             vec3f P) const;
#endif
      const float   *scalars;
      /*! per-cell active mask (parallel to `scalars`): 0 where the cell is
          refined by a finer block (covered) and must be excluded from the basis
          sum, 1 otherwise. Null => treat every cell as active. Set by the
          sampler (which owns the bvh used to detect coverage). */
      const uint8_t *active = nullptr;
      struct {
        const vec3i *origins;
        const vec3i *dims;
        const int   *levels;
        const uint64_t *offsets;
      } perBlock;
      struct {
        const int   *refinements;
      } perLevel;
      int numBlocks;
      float finestCellSize; // smallest cell size present (for iso octant sampling)
    };

    BlockStructuredField(Context *context,
                         const DevGroup::SP &devices);
    virtual ~BlockStructuredField() override;
    
    DD getDD(Device *device);
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool setData(const std::string &member,
                 const std::shared_ptr<Data> &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    
    MCGrid::SP buildMCs() override;
    
    /*! computes, on specified device, the array of bounding box and
        value ranges for cubql bvh consturction; one box and one value
        range per each block */
    void computeElementBBs(Device *device,
                           box3f *d_primBounds,
                           range1f *d_primRanges);
    
    VolumeAccel::SP createAccel(Volume *volume) override;

    /*! creates a macro-cell accelerated iso-surface accel for this AMR
        field, reusing the same cuBQL sampler and macro-cell grid the
        volume path uses */
    IsoSurfaceAccel::SP createIsoAccel(IsoSurface *isoSurface) override;

    struct {
      PODData::SP/*3i*/ origins    = 0;
      PODData::SP/*3i*/ dims       = 0;
      PODData::SP/*1i*/ levels     = 0;
      PODData::SP/*1l*/ offsets    = 0;
    } perBlock;
    struct {
      PODData::SP/*1i*/ refinements = 0;
    } perLevel;
    PODData::SP/*1f*/   scalars     = 0;
    int                 numBlocks   = 0;
    float               finestCellSize = 0.f;
  };


  struct Block
  {
#if RTC_DEVICE_CODE
    static
    inline __rtc_device Block getFrom(const BlockStructuredField::DD &dd, int blockID, bool dbg=false);
    
    inline __rtc_device float getScalar(const vec3i cellID) const;
    inline __rtc_device box3f cellBounds(const vec3i cellID) const;
    inline __rtc_device box3f getDomain() const;
    inline __rtc_device range1f getValueRange() const;
#endif
    vec3i origin;
    vec3i dims;
    int   level;
    float cellSize;
    const float *scalars;
  };
  
  
#if RTC_DEVICE_CODE
  /* compute basis function contribution of given block at point P, and add
     that to 'sumWeightedValues' and 'sumWeights'. returns true if P is inside
     the block *filter domain*, false if outside (in which case the out params
     are not defined) */
  inline __rtc_device
  bool BlockStructuredField::DD::addBasisFunctions(float &sumWeightedValues,
                                                   float &sumWeights,
                                                   uint32_t bid,
                                                   vec3f P) const
  {
    const auto block = Block::getFrom(*this,bid);
    const box3f domain = block.getDomain();

    if (!domain.contains(P)) return false;

    const vec3f cellCenter000 = domain.lower+vec3f(block.cellSize);
    const vec3f localPos
      = (P-cellCenter000) / block.cellSize;

    vec3f floor_localPos(floorf(localPos.x),
                         floorf(localPos.y),
                         floorf(localPos.z));
    vec3i idx_lo   = vec3i(floor_localPos);
    idx_lo = max(vec3i(-1), idx_lo);
    const vec3i idx_hi   = idx_lo + vec3i(1);
    const vec3f frac     = localPos - floor_localPos;
    const vec3f neg_frac = vec3f(1.f) - frac;

    // this block's base offset into the global scalar / active arrays
    const uint64_t goff = (uint64_t)(block.scalars - scalars);
    auto addCell = [&](int ix, int iy, int iz, float w) {
      if (ix < 0 || iy < 0 || iz < 0 ||
          ix >= block.dims.x || iy >= block.dims.y || iz >= block.dims.z)
        return;
      const uint64_t lidx
        = (uint64_t)ix + block.dims.x*((uint64_t)iy + block.dims.y*(uint64_t)iz);
      // skip cells refined by a finer block: they are the coarse-under-fine
      // cells that would otherwise over-blend the reconstruction
      if (active && !active[goff + lidx]) return;
      sumWeights        += w;
      sumWeightedValues += w * block.scalars[lidx];
    };

    addCell(idx_lo.x, idx_lo.y, idx_lo.z, neg_frac.x*neg_frac.y*neg_frac.z);
    addCell(idx_hi.x, idx_lo.y, idx_lo.z,     frac.x*neg_frac.y*neg_frac.z);
    addCell(idx_lo.x, idx_hi.y, idx_lo.z, neg_frac.x*    frac.y*neg_frac.z);
    addCell(idx_hi.x, idx_hi.y, idx_lo.z,     frac.x*    frac.y*neg_frac.z);
    addCell(idx_lo.x, idx_lo.y, idx_hi.z, neg_frac.x*neg_frac.y*    frac.z);
    addCell(idx_hi.x, idx_lo.y, idx_hi.z,     frac.x*neg_frac.y*    frac.z);
    addCell(idx_lo.x, idx_hi.y, idx_hi.z, neg_frac.x*    frac.y*    frac.z);
    addCell(idx_hi.x, idx_hi.y, idx_hi.z,     frac.x*    frac.y*    frac.z);
    return true;
  }

  /*! analytic-gradient sibling of addBasisFunctions: accumulates, for this
      block's active cells, the tent weight H_C and its per-axis partials into
      'acc'. Each tent is the trilinear product w = wx*wy*wz over the local cell
      coordinate; its x-partial is (dwx/dx)*wy*wz with dwx/dx = +-1/cellWidth (sign
      = +1 for the upper cell along that axis, -1 for the lower). Summed across all
      blocks/levels this gives the exact gradient of the reconstruction B=N/D. */
  inline __rtc_device
  void BlockStructuredField::DD::addBasisFunctionsGrad(BasisGrad &acc,
                                                       uint32_t bid,
                                                       vec3f P) const
  {
    const auto block = Block::getFrom(*this,bid);
    const box3f domain = block.getDomain();
    if (!domain.contains(P)) return;

    const vec3f cellCenter000 = domain.lower+vec3f(block.cellSize);
    const vec3f localPos = (P-cellCenter000) / block.cellSize;
    vec3f floor_localPos(floorf(localPos.x),
                         floorf(localPos.y),
                         floorf(localPos.z));
    vec3i idx_lo = max(vec3i(-1), vec3i(floor_localPos));
    const vec3i idx_hi   = idx_lo + vec3i(1);
    const vec3f frac     = localPos - floor_localPos;
    const vec3f neg_frac = vec3f(1.f) - frac;
    const float invcw    = 1.f / block.cellSize;   // d(localPos)/dP per axis

    const uint64_t goff = (uint64_t)(block.scalars - scalars);
    // c encodes the corner: bit0=x, bit1=y, bit2=z (0 -> lower cell, 1 -> upper)
    for (int c=0;c<8;c++) {
      const int bx=(c&1), by=((c>>1)&1), bz=((c>>2)&1);
      const int ix = bx ? idx_hi.x : idx_lo.x;
      const int iy = by ? idx_hi.y : idx_lo.y;
      const int iz = bz ? idx_hi.z : idx_lo.z;
      if (ix < 0 || iy < 0 || iz < 0 ||
          ix >= block.dims.x || iy >= block.dims.y || iz >= block.dims.z)
        continue;
      const uint64_t lidx
        = (uint64_t)ix + block.dims.x*((uint64_t)iy + block.dims.y*(uint64_t)iz);
      if (active && !active[goff + lidx]) continue;

      const float wx = bx ? frac.x : neg_frac.x;
      const float wy = by ? frac.y : neg_frac.y;
      const float wz = bz ? frac.z : neg_frac.z;
      const float sx = bx ? invcw : -invcw;        // d(wx)/dP.x
      const float sy = by ? invcw : -invcw;
      const float sz = bz ? invcw : -invcw;
      const float v  = block.scalars[lidx];

      const float w  = wx*wy*wz;
      const float dHx = sx*wy*wz, dHy = wx*sy*wz, dHz = wx*wy*sz;
      acc.D  += w;    acc.N  += w*v;
      acc.Dx += dHx;  acc.Nx += dHx*v;
      acc.Dy += dHy;  acc.Ny += dHy*v;
      acc.Dz += dHz;  acc.Nz += dHz*v;
    }
  }

  inline __rtc_device
  float Block::getScalar(const vec3i cellID) const
  {
    const int idx
      = 
      + cellID.x
      + cellID.y * dims.x
      + cellID.z * dims.x*dims.y;
    return scalars[idx];
  }

  inline __rtc_device
  box3f Block::cellBounds(const vec3i cellID) const
  {
    box3f cb;
    cb.lower = (vec3f(origin+cellID)-.5f)*cellSize;
    cb.upper = cb.lower + 2.f*cellSize;
    return cb;
  }

  inline __rtc_device
  range1f Block::getValueRange() const
  {
    range1f range;
    for (int i=0;i<dims.x*dims.y*dims.z;i++)
      range.extend(scalars[i]);
    return range;
  }
  
  inline __rtc_device
  box3f Block::getDomain() const
  {
    box3f cb;
    cb.lower = (vec3f(origin)-.5f)*cellSize;
    cb.upper = (vec3f(origin+dims)+.5f)*cellSize;
    return cb;
  }

  inline __rtc_device
  Block Block::getFrom(const BlockStructuredField::DD &dd, int blockID, bool dbg)
  {
    Block block;
    block.origin   = dd.perBlock.origins[blockID];
    block.dims     = dd.perBlock.dims[blockID];
    block.level    = dd.perBlock.levels[blockID];
    block.cellSize = (int)(powf((float)dd.perLevel.refinements[block.level],
                                (float)block.level));
    block.scalars  = dd.scalars+dd.perBlock.offsets[blockID];
    return block;
  }
#endif
}
