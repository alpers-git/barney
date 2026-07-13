// SPDX-FileCopyrightText:
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier:
// Apache-2.0


#pragma once

#include "barney/amr/BlockStructuredField.h"
#include "barney/volume/MCAccelerator.h"
#include "barney/common/CuBQL.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace BARNEY_NS {

  struct BlockStructuredField;

  /*! finest leaf cell covering a point: value, world cell size, center, and the
      owning block's context so neighbours in the same block need no bvh query */
  struct AMRLeaf {
    float value;
    float cellSize;
    vec3f center;
    bool  found;
    vec3i origin;
    vec3i dims;
    vec3i cellID;
    const float *scalars;
  };

  /*! a block structured amr scalar field, with a CuBQL bvh sampler */
  struct BlockStructuredCuBQLSampler : public ScalarFieldSampler {
    enum { BVH_WIDTH = 4 };
    using bvh_t  = cuBQL::WideBVH<float,3,BVH_WIDTH>;
    using node_t = typename bvh_t::Node;

    struct DD : public BlockStructuredField::DD {
#if RTC_DEVICE_CODE
      /*! State-of-the-art crack-free AMR reconstruction (ExaBricks): a
          normalized sum of per-cell tent basis functions over the ACTIVE (leaf)
          cells only. Smooth, C0-continuous, and one bvh query per sample. The
          coarse cells Chombo stores *under* refined regions are excluded via the
          per-cell `active` mask, which is what previously caused boundary bands. */
      inline __rtc_device float sample(vec3f P, bool dbg = false) const;
      /*! reconstructs the value at P *and* its analytic object-space gradient
          (ExaBricks eq. 3) in the same single bvh query - used for iso shading
          normals in place of a 7-tap central difference. Returns NAN (grad 0)
          outside the volume. */
      inline __rtc_device float sampleGrad(vec3f P, vec3f &grad) const;
      /*! world-space size of the finest cell whose block covers P, or 0 in a
          coverage gap; used to scale the iso-surface shading-normal step */
      inline __rtc_device float localCellSize(vec3f P) const;
      /*! crack-free octant reconstruction (iso path only) */
      inline __rtc_device float octantSample(vec3f P) const;
      /*! one-query dual-cell reconstruction (iso path): locates the finest block
          covering P and, when P's whole dual cell (the cube of 8 surrounding
          cell centres) lies inside that block, returns the trilinear value plus
          its exact object-space gradient. Sets found=false at block / level
          boundaries so the caller can fall back to the crack-free octant path. */
      inline __rtc_device float dualValueGrad(vec3f P, vec3f &grad, bool &found) const;
      /*! cheap one-query reconstruction for the crossing march: finest block
          trilinear, clamped at block edges - no neighbour or octant queries, so
          every march sample costs a single bvh traversal. Boundary accuracy is
          approximate; the exact octant path is reserved for the shading normal. */
      inline __rtc_device float dualClamped(vec3f P) const;
      inline __rtc_device AMRLeaf cellAt(vec3f q) const;
      inline __rtc_device AMRLeaf neighborValue(const AMRLeaf &leaf, vec3i delta) const;
      inline __rtc_device float finestLevelLerp(vec3f vpos, float hf) const;
#endif
      bvh_t bvh;
    };
    DD getDD(Device *device);

    /*! per-device data - parent store the bs-amr field, we just store the
      bvh nodes and the per-cell active (non-covered) mask */
    struct PLD {
      bvh_t bvh = { 0,0 };
      uint8_t *active = nullptr;   //!< 1 per cell: 0 if refined by a finer block
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;

    BlockStructuredCuBQLSampler(BlockStructuredField *mesh);
    
    /*! builds the string that allows for properly matching optix
      device progs for this type */
    inline static std::string typeName() { return "BlockStructured_CuBQL"; }

    void build() override;

    BlockStructuredField *const field;
    const DevGroup::SP devices;
  };
  
  struct BlockStructuredSamplerPTD {
    inline __rtc_device BlockStructuredSamplerPTD(const BlockStructuredCuBQLSampler::DD *field)
      : field(field)
    {}
#if RTC_DEVICE_CODE
    inline __rtc_device void visitBrick(vec3f P, int primID)
    {
      field->addBasisFunctions(sumWeightedValues,sumWeights,primID,P);
    }
#endif
    const BlockStructuredCuBQLSampler::DD *const field;
    
    float sumWeights = 0.f;
    float sumWeightedValues = 0.f;
  };
  
#if RTC_DEVICE_CODE
  /*! State-of-the-art crack-free AMR reconstruction: the ExaBricks tent-basis
      method, restricted to ACTIVE (leaf) cells.

      Each cell carries a trilinear tent basis of its own width, and the value at
      P is the normalized sum over all cells whose tent covers P,
      sum(w_i v_i)/sum(w_i). This is a ratio of continuous functions, hence
      C0-continuous everywhere - no cracks at level *or* same-level boundaries -
      and smooth within every level (proper tent interpolation), all in a single
      bvh query.

      The one fix over the old tent-basis is that the per-cell `active` mask (built
      once, marking a cell covered when a finer block refines it) excludes the
      coarse cells Chombo stores *underneath* refined regions. Those covered cells
      were being blended in, over-smoothing fine detail and bleeding covered-cell
      values - the opaque bands seen at boundaries. With them gone, only the
      genuine half-cell overlap at level transitions remains, which is exactly what
      makes the basis crack-free. */
  inline __rtc_device
  float BlockStructuredCuBQLSampler::DD::sample(vec3f P, bool dbg) const
  {
    BlockStructuredSamplerPTD ptd(this);

    auto lambda = [&](const uint32_t primID) -> int {
      ptd.visitBrick(P,primID);
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    cuBQL::box3f box; box.lower = box.upper = (const cuBQL::vec3f &)P;
    cuBQL::fixedBoxQuery::forEachPrim(lambda,bvh,box);
    return ptd.sumWeights == 0.f ? NAN : (ptd.sumWeightedValues  / ptd.sumWeights);
  }

  /*! value + analytic gradient of the active-cell basis reconstruction, in one
      bvh query (ExaBricks, Wald et al. 2020, eq. 3). Cheaper and higher quality
      than clamped central differences: no extra traversals, and the gradient is
      the true derivative of B=N/D via the quotient rule,
      dB/dq = (D*dN/dq - N*dD/dq)/D^2. B=N/D is C0 but not C1, so this can differ
      slightly across cell/level boundaries - the intended real-time tradeoff. */
  inline __rtc_device
  float BlockStructuredCuBQLSampler::DD::sampleGrad(vec3f P, vec3f &grad) const
  {
    BasisGrad acc;
    auto lambda = [&](const uint32_t primID) -> int {
      addBasisFunctionsGrad(acc,primID,P);
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    cuBQL::box3f box; box.lower = box.upper = (const cuBQL::vec3f &)P;
    cuBQL::fixedBoxQuery::forEachPrim(lambda,bvh,box);
    if (acc.D == 0.f) { grad = vec3f(0.f); return NAN; }
    const float invD = 1.f/acc.D;
    grad = vec3f(acc.D*acc.Nx - acc.N*acc.Dx,
                 acc.D*acc.Ny - acc.N*acc.Dy,
                 acc.D*acc.Nz - acc.N*acc.Dz) * (invD*invD);
    return acc.N*invD;
  }

  inline __rtc_device
  float BlockStructuredCuBQLSampler::DD::localCellSize(vec3f P) const
  {
    // Blocks overlap at boundaries and across levels; the finest (smallest
    // cellSize) block covering P sets the local resolution.
    float best = 0.f;
    auto lambda = [&](const uint32_t primID) -> int {
      const auto block = Block::getFrom(*this,primID);
      if (block.getDomain().contains(P))
        if (best == 0.f || block.cellSize < best)
          best = block.cellSize;
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    cuBQL::box3f box; box.lower = box.upper = (const cuBQL::vec3f &)P;
    cuBQL::fixedBoxQuery::forEachPrim(lambda,bvh,box);
    return best;
  }

  /*! ADL overload of the iso-surface step hook (see MCAccelerator.h): give the
      shading-normal finite difference a step matched to the local AMR cell. */
  inline __rtc_device
  float isoLocalStep(const BlockStructuredCuBQLSampler::DD &s, vec3f P, float fallback)
  {
    const float cs = s.localCellSize(P);
    return cs > 0.f ? cs : fallback;
  }

  inline __rtc_device
  AMRLeaf BlockStructuredCuBQLSampler::DD::cellAt(vec3f q) const
  {
    AMRLeaf best; best.found = false; best.cellSize = 1e30f; best.value = 0.f;
    auto lambda = [&](const uint32_t primID) -> int {
      const auto block = Block::getFrom(*this,primID);
      const float cs = block.cellSize;
      if (cs >= best.cellSize) return CUBQL_CONTINUE_TRAVERSAL;
      const vec3f lo = vec3f(block.origin) * cs;
      const vec3f hi = vec3f(block.origin + block.dims) * cs;
      if (q.x < lo.x || q.y < lo.y || q.z < lo.z ||
          q.x >= hi.x || q.y >= hi.y || q.z >= hi.z)
        return CUBQL_CONTINUE_TRAVERSAL;
      vec3i cell = vec3i(int(floorf(q.x/cs)),int(floorf(q.y/cs)),int(floorf(q.z/cs)))
        - block.origin;
      cell = max(vec3i(0),min(block.dims - vec3i(1),cell));
      best.value    = block.getScalar(cell);
      best.cellSize = cs;
      best.center   = (vec3f(block.origin + cell) + vec3f(0.5f)) * cs;
      best.found    = true;
      best.origin   = block.origin;
      best.dims     = block.dims;
      best.cellID   = cell;
      best.scalars  = block.scalars;
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    cuBQL::box3f box; box.lower = box.upper = (const cuBQL::vec3f &)q;
    cuBQL::fixedBoxQuery::forEachPrim(lambda,bvh,box);
    return best;
  }

  /*! neighbour cell 'delta' cells away: an O(1) in-block read when it stays
      inside the leaf's block, else a bvh lookup */
  inline __rtc_device
  AMRLeaf BlockStructuredCuBQLSampler::DD::neighborValue(const AMRLeaf &leaf,
                                                        vec3i delta) const
  {
    const vec3i nb = leaf.cellID + delta;
    if (nb.x>=0 && nb.y>=0 && nb.z>=0 &&
        nb.x<leaf.dims.x && nb.y<leaf.dims.y && nb.z<leaf.dims.z) {
      AMRLeaf r = leaf;
      r.cellID = nb;
      r.value  = leaf.scalars[nb.x + leaf.dims.x*(nb.y + leaf.dims.y*nb.z)];
      r.center = (vec3f(leaf.origin + nb) + vec3f(0.5f)) * leaf.cellSize;
      return r;
    }
    return cellAt(leaf.center + leaf.cellSize*vec3f(delta));
  }

  inline __rtc_device
  float BlockStructuredCuBQLSampler::DD::finestLevelLerp(vec3f vpos, float hf) const
  {
    const vec3f g = vpos * (1.f/hf) - vec3f(0.5f);
    const vec3i base(int(floorf(g.x)),int(floorf(g.y)),int(floorf(g.z)));
    float sum = 0.f; int cnt = 0;
    for (int dz=0;dz<2;dz++)
      for (int dy=0;dy<2;dy++)
        for (int dx=0;dx<2;dx++) {
          const vec3f node = (vec3f(base + vec3i(dx,dy,dz)) + vec3f(0.5f)) * hf;
          const AMRLeaf cell = cellAt(node);
          if (cell.found) { sum += cell.value; ++cnt; }
        }
    return cnt ? sum/float(cnt) : NAN;
  }

  inline __rtc_device
  float BlockStructuredCuBQLSampler::DD::octantSample(vec3f P) const
  {
    const AMRLeaf leaf = cellAt(P);
    if (!leaf.found) return NAN;
    const float h = leaf.cellSize;
    const vec3f c = leaf.center;
    const vec3f s(P.x>=c.x?1.f:-1.f, P.y>=c.y?1.f:-1.f, P.z>=c.z?1.f:-1.f);
    const vec3i si(int(s.x),int(s.y),int(s.z));
    // in-block neighbour reads are only valid when no finer block can overlap,
    // i.e. the leaf is at the finest level; otherwise fall back to bvh lookups
    const bool finest = h <= finestCellSize*1.0001f;

    // 2x2x2 dual neighbourhood in the octant's direction
    AMRLeaf N[2][2][2];
    for (int dz=0;dz<2;dz++)
      for (int dy=0;dy<2;dy++)
        for (int dx=0;dx<2;dx++) {
          if (!(dx||dy||dz)) { N[dz][dy][dx] = leaf; continue; }
          const vec3i d(dx*si.x,dy*si.y,dz*si.z);
          N[dz][dy][dx] = finest ? neighborValue(leaf,d)
                                 : cellAt(c + h*vec3f(d));
        }

    // octant vertex values, stitched to the coarser side at level boundaries
    float V[2][2][2];
    for (int bz=0;bz<2;bz++)
      for (int by=0;by<2;by++)
        for (int bx=0;bx<2;bx++) {
          float sum = 0.f; int cnt = 0; float minCS = 1e30f; bool boundary = false;
          for (int dz=0;dz<=bz;dz++)
            for (int dy=0;dy<=by;dy++)
              for (int dx=0;dx<=bx;dx++) {
                const AMRLeaf &cell = N[dz][dy][dx];
                if (!cell.found) { boundary = true; continue; }
                sum += cell.value; ++cnt;
                if (cell.cellSize != h) boundary = true;
                if (cell.cellSize < minCS) minCS = cell.cellSize;
              }
          if (!boundary && cnt)
            V[bz][by][bx] = sum/float(cnt);
          else {
            const vec3f vpos = c + 0.5f*h*vec3f(bx*s.x,by*s.y,bz*s.z);
            V[bz][by][bx] = finestLevelLerp(vpos, minCS<1e30f?minCS:h);
          }
        }

    const vec3f t((P.x-c.x)/(s.x*0.5f*h),
                  (P.y-c.y)/(s.y*0.5f*h),
                  (P.z-c.z)/(s.z*0.5f*h));
    const vec3f tt(min(1.f,max(0.f,t.x)),min(1.f,max(0.f,t.y)),min(1.f,max(0.f,t.z)));
    float result = 0.f;
    for (int bz=0;bz<2;bz++)
      for (int by=0;by<2;by++)
        for (int bx=0;bx<2;bx++)
          result += (bx?tt.x:1.f-tt.x)*(by?tt.y:1.f-tt.y)*(bz?tt.z:1.f-tt.z)
            * V[bz][by][bx];
    return result;
  }

  inline __rtc_device
  float BlockStructuredCuBQLSampler::DD::dualValueGrad(vec3f P,
                                                      vec3f &grad,
                                                      bool &found) const
  {
    found = false; grad = vec3f(NAN);
    // finest cell size among blocks whose cell domain covers P, and the finest
    // block that also fully contains P's dual cell (its 8 surrounding centres)
    float finestDomainCS = 1e30f;
    float dualCS = 1e30f, dualV = NAN; vec3f dualG(0.f); bool dualOK = false;
    auto lambda = [&](const uint32_t primID) -> int {
      const auto b = Block::getFrom(*this,primID);
      const float cs = b.cellSize;
      const vec3f domLo = vec3f(b.origin) * cs;
      const vec3f domHi = vec3f(b.origin + b.dims) * cs;
      if (P.x>=domLo.x && P.y>=domLo.y && P.z>=domLo.z &&
          P.x< domHi.x && P.y< domHi.y && P.z< domHi.z)
        if (cs < finestDomainCS) finestDomainCS = cs;
      if (cs < dualCS) {
        // continuous cell-centre coordinate: cell centres sit at integer g
        const vec3f g = P*(1.f/cs) - vec3f(b.origin) - vec3f(0.5f);
        const vec3i lo(int(floorf(g.x)),int(floorf(g.y)),int(floorf(g.z)));
        if (lo.x>=0 && lo.y>=0 && lo.z>=0 &&
            lo.x+1<b.dims.x && lo.y+1<b.dims.y && lo.z+1<b.dims.z) {
          float V[2][2][2];
          for (int dz=0;dz<2;dz++)
            for (int dy=0;dy<2;dy++)
              for (int dx=0;dx<2;dx++)
                V[dz][dy][dx] = b.getScalar(lo+vec3i(dx,dy,dz));
          const vec3f t = g - vec3f(lo);
          const float c00=V[0][0][0]+(V[0][0][1]-V[0][0][0])*t.x;
          const float c01=V[0][1][0]+(V[0][1][1]-V[0][1][0])*t.x;
          const float c10=V[1][0][0]+(V[1][0][1]-V[1][0][0])*t.x;
          const float c11=V[1][1][0]+(V[1][1][1]-V[1][1][0])*t.x;
          const float c0=c00+(c01-c00)*t.y;
          const float c1=c10+(c11-c10)*t.y;
          dualV = c0+(c1-c0)*t.z;
          const float gx=
             (1.f-t.y)*(1.f-t.z)*(V[0][0][1]-V[0][0][0])
            +(     t.y)*(1.f-t.z)*(V[0][1][1]-V[0][1][0])
            +(1.f-t.y)*(     t.z)*(V[1][0][1]-V[1][0][0])
            +(     t.y)*(     t.z)*(V[1][1][1]-V[1][1][0]);
          const float gy=
             (1.f-t.x)*(1.f-t.z)*(V[0][1][0]-V[0][0][0])
            +(     t.x)*(1.f-t.z)*(V[0][1][1]-V[0][0][1])
            +(1.f-t.x)*(     t.z)*(V[1][1][0]-V[1][0][0])
            +(     t.x)*(     t.z)*(V[1][1][1]-V[1][0][1]);
          const float gz=
             (1.f-t.x)*(1.f-t.y)*(V[1][0][0]-V[0][0][0])
            +(     t.x)*(1.f-t.y)*(V[1][0][1]-V[0][0][1])
            +(1.f-t.x)*(     t.y)*(V[1][1][0]-V[0][1][0])
            +(     t.x)*(     t.y)*(V[1][1][1]-V[0][1][1]);
          dualG = vec3f(gx,gy,gz)*(1.f/cs);
          dualCS = cs; dualOK = true;
        }
      }
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    cuBQL::box3f box; box.lower = box.upper = (const cuBQL::vec3f &)P;
    cuBQL::fixedBoxQuery::forEachPrim(lambda,bvh,box);
    // only trust the in-block reconstruction when it came from the *finest*
    // block covering P; otherwise P is at that block's boundary and a coarser
    // block would crack, so let the caller stitch with the octant method
    if (dualOK && dualCS <= finestDomainCS*1.0001f) {
      grad = dualG; found = true; return dualV;
    }
    return NAN;
  }

  inline __rtc_device
  float BlockStructuredCuBQLSampler::DD::dualClamped(vec3f P) const
  {
    float bestCS = 1e30f, outV = NAN;
    auto lambda = [&](const uint32_t primID) -> int {
      const auto b = Block::getFrom(*this,primID);
      const float cs = b.cellSize;
      if (cs >= bestCS) return CUBQL_CONTINUE_TRAVERSAL;
      if (b.dims.x<2 || b.dims.y<2 || b.dims.z<2)
        return CUBQL_CONTINUE_TRAVERSAL;
      const vec3f domLo = vec3f(b.origin) * cs;
      const vec3f domHi = vec3f(b.origin + b.dims) * cs;
      if (P.x<domLo.x || P.y<domLo.y || P.z<domLo.z ||
          P.x>=domHi.x || P.y>=domHi.y || P.z>=domHi.z)
        return CUBQL_CONTINUE_TRAVERSAL;
      const vec3f g = P*(1.f/cs) - vec3f(b.origin) - vec3f(0.5f);
      vec3i lo(int(floorf(g.x)),int(floorf(g.y)),int(floorf(g.z)));
      lo = max(vec3i(0),min(b.dims-vec3i(2),lo));   // keep lo..lo+1 in range
      const vec3f gl = g - vec3f(lo);
      const vec3f t(min(1.f,max(0.f,gl.x)),
                    min(1.f,max(0.f,gl.y)),
                    min(1.f,max(0.f,gl.z)));
      float V[2][2][2];
      for (int dz=0;dz<2;dz++)
        for (int dy=0;dy<2;dy++)
          for (int dx=0;dx<2;dx++)
            V[dz][dy][dx] = b.getScalar(lo+vec3i(dx,dy,dz));
      const float c00=V[0][0][0]+(V[0][0][1]-V[0][0][0])*t.x;
      const float c01=V[0][1][0]+(V[0][1][1]-V[0][1][0])*t.x;
      const float c10=V[1][0][0]+(V[1][0][1]-V[1][0][0])*t.x;
      const float c11=V[1][1][0]+(V[1][1][1]-V[1][1][0])*t.x;
      const float c0=c00+(c01-c00)*t.y;
      const float c1=c10+(c11-c10)*t.y;
      outV = c0+(c1-c0)*t.z; bestCS = cs;
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    cuBQL::box3f box; box.lower = box.upper = (const cuBQL::vec3f &)P;
    cuBQL::fixedBoxQuery::forEachPrim(lambda,bvh,box);
    return outV;
  }

  /*! ADL overload of the iso-surface sample hook (see MCAccelerator.h): the
      crossing march uses the cheap one-query clamped reconstruction; the exact
      dual-cell + crack-free octant path is kept for the shading gradient below.
      Volume rendering keeps using DD::sample(). */
  inline __rtc_device
  float isoSample(const BlockStructuredCuBQLSampler::DD &s, vec3f P)
  { return s.dualClamped(P); }

  /*! ADL overload of the iso-surface gradient hook (see MCAccelerator.h): the
      dual-cell reconstruction is trilinear, so its gradient is analytic and free
      from the same lookup. Falls back to the octant value with a NaN gradient at
      boundaries so the caller finite-differences only there. */
  inline __rtc_device
  float isoSampleGrad(const BlockStructuredCuBQLSampler::DD &s, vec3f P, vec3f &grad)
  {
    bool found;
    const float v = s.dualValueGrad(P,grad,found);
    if (found) return v;
    grad = vec3f(NAN);
    return s.octantSample(P);
  }

  /*! AMR override of the iso analytic-gradient customization point declared in
      MCAccelerator.h. */
  inline __rtc_device
  bool isoAnalyticGrad(const BlockStructuredCuBQLSampler::DD &/*s*/,
                       vec3f /*P*/, float &/*value*/, vec3f &/*grad*/)
  {
    return false;
  }
#endif
}


