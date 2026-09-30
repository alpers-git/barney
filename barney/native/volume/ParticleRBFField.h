// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Object.h"
#include "native/ModelSlot.h"
#include "native/volume/MCAccelerator.h"
#include <limits>
#include <vector>

namespace BARNEY_NS {
  namespace native {

    struct ParticleRBFSampler;

    /*! largest polynomial degree a blob may carry; sizes the per-blob power
        table, which lives in registers. l<=3 on both sides of a bond would
        need 6, and f carries 2.4% of the current, so 6 covers everything the
        builder can emit. */
    enum { RBF_MAX_DEGREE = 6 };

    /*! number of coefficients in a dense polynomial of degree <= d */
    inline __rtc_both int rbf_numCoeffs(int d)
    { return (d+1)*(d+2)*(d+3)/6; }

    /*! position of monomial x^kx y^ky z^kz in the canonical order the dense
        coefficient array uses: kx descending, then ky descending, then kz
        descending - the order a nested Horner evaluation consumes them in, so
        evalPoly() reads them strictly sequentially and needs no powers
        tabulated and no per-term multiplies. */
    inline __rtc_both int rbf_monomialSlot(int kx, int ky, int kz, int D)
    {
      const int M = D - kx;
      return (D-kx)*(D-kx+1)*(D-kx+2)/6      // all slots with a larger x power
        +    (M-ky)*(M-ky+1)/2               // ... with this x, larger y power
        +    (M-ky-kz);                      // ... and this y, larger z power
    }

    /*! a scalar field made of signed, polynomial-modulated radial basis
        functions - i.e. each primitive is

          coeff * x^px * y^py * z^pz * exp(-gamma*r^2),  r = |P-center|

        evaluated in the particle's local frame, and the field is the *sum*
        over all primitives whose support contains P. this covers gaussian
        type orbitals (a cartesian monomial times a gaussian is exactly what a
        contracted GTO primitive is), and - via the gaussian product theorem -
        also products of two such orbitals, which is why orbital *pair*
        interactions need no separate primitive type.

        unlike the other scalar field types this one is *signed*: the sum can
        be negative, so macro-cell ranges have to be conservative over
        cancellation, and the transfer function domain is expected to straddle
        zero. */
    struct ParticleRBFField : public ScalarField
    {
      typedef std::shared_ptr<ParticleRBFField> SP;

      /*! how many macro-cells along the longest world-bounds axis. this is far
          coarser than the other field types on purpose: one sample here costs a
          range query over hundreds of primitives, so the cost is dominated by
          how many cells a ray steps through, not by how tightly they bound the
          field - and each cell is re-sampled internally anyway. measured on the
          orbital data, dropping 256 -> 24 was 4x faster for 0.3% of pixels
          differing by more than 8/255. */
      enum { DEFAULT_MC_GRID_SIZE = 32 };

      /*! device-side selection state. 'groups' are the logical units a user
          selects (one orbital, or one orbital *pair* interaction); many
          primitives share a group, so selection is a per-group byte lookup
          rather than anything per-primitive. */
      struct Filter {
        /*! one byte per group, 0 == hidden; null means 'all visible' */
        const uint8_t *groupEnabled;
        /*! per-group scalar the continuous filter tests (e.g. |J_ij|) */
        const float   *groupValues;
        /*! groups whose |value| falls outside this are hidden */
        range1f        valueRange;
        int            numGroups;

#if RTC_DEVICE_CODE
        inline __rtc_device bool passes(uint32_t groupID) const
        {
          if (groupEnabled && !groupEnabled[groupID]) return false;
          if (groupValues) {
            const float a = fabsf(groupValues[groupID]);
            if (a < valueRange.lower || a > valueRange.upper) return false;
          }
          return true;
        }
#endif
      };

      /*! how a vector field is reduced to the scalar the transfer function
          and the iso-surfaces see. only meaningful when 'directions' is set. */
      enum VecMode { VEC_MAGNITUDE=0, VEC_X=1, VEC_Y=2, VEC_Z=3 };

      struct DD : public ScalarField::DD {
#if RTC_DEVICE_CODE
        /*! contribution of a single primitive at P; 0 if P is outside the
            primitive's cutoff or the primitive's group is filtered out */
        inline __rtc_device float evalParticle(int pid, vec3f P) const;

        /*! scalar part of a polynomial blob: exp(-gamma r^2) times the sum of
            the enabled channels' polynomials. this is f_AB(r) - the bond's
            current profile - with the bond direction still factored out. */
        inline __rtc_device float evalPoly(int pid, vec3f d, float r2) const;

        /*! vector contribution u_AB * f_AB(P) of one blob. |j| needs the sum
            over blobs before the length is taken, so the sampler accumulates
            these and reduces once. */
        inline __rtc_device vec3f evalParticleVec(int pid, vec3f P) const;

        /*! world-space support of one primitive, as used for both the cuBQL
            bvh and the macro-cell rasterisation */
        inline __rtc_device box3f particleBounds(int pid) const;

        /*! conservative bound on |contribution| of primitive 'pid' anywhere
            within distance range [rLo,rHi] of its center */
        inline __rtc_device float particleBound(int pid, float rLo, float rHi) const;

        /*! true if the primitive's angular part cannot change sign, i.e. all
            monomial powers are even - lets the majorant keep one-sided */
        inline __rtc_device bool signDefinite(int pid) const;
#endif
        const vec3f    *centers;
        const float    *coeffs;
        const float    *gammas;
        const float    *cutoffs;
        /*! three 8-bit monomial powers (px,py,pz) packed low-to-high; null
            means every primitive is isotropic */
        const uint32_t *monomials;
        /*! group each primitive belongs to; null means one implicit group */
        const uint32_t *groupIDs;
        int             numParticles;

        // ---- polynomial-blob mode (null 'polyOffsets' means plain rbf) ----
        /*! csr into polyCoeffs/polyMonomials, numChannels entries per
            primitive: channel c of primitive p spans
            [polyOffsets[p*numChannels+c], polyOffsets[p*numChannels+c+1]).
            One primitive is then one gaussian carrying a whole polynomial per
            orbital channel, instead of one monomial term - which is what keeps
            a bond to nexp_A*nexp_B primitives rather than millions. */
        const uint32_t *polyOffsets;
        const float    *polyCoeffs;
        const uint32_t *polyMonomials;
        /*! the enabled channels summed into one dense polynomial per blob,
            'polyStride' coefficients each, in the canonical monomial order
            rbf_monomialSlot() defines. Rebuilt only when the channel mask
            changes, which turns a per-sample sum over 9 channels' worth of
            scattered csr terms into one contiguous dot product - the channels
            are a *selection*, and a selection should cost nothing to render. */
        const float    *polyDense;
        int             polyStride;
        /*! unit bond direction per *group*; when set the field is a vector
            field and 'vecMode' says how it is reduced to a scalar */
        const vec3f    *directions;
        int             numChannels;
        /*! bit c enables channel c; channels are switched here rather than by
            rebuilding, because every channel shares one blob's geometry */
        uint32_t        channelMask;
        int             vecMode;
        int             polyMaxDegree;
        /*! the primitives the current filter keeps, as indices into the arrays
            above. the bvh is built over *these*, so hiding groups shrinks what
            a range query has to walk instead of only zeroing what it finds. */
        const uint32_t *activeIDs;
        int             numActive;
        Filter          filter;
      };

      ParticleRBFField(Context *context, const DevGroup::SP &devices);
      virtual ~ParticleRBFField() override;

      /*! pretty-printer for printf-debugging */
      std::string toString() const override;

      DD getDD(Device *device);

      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      void commit() override;
      bool setData(const std::string &member,
                   const std::shared_ptr<Data> &value) override;
      bool set1f(const std::string &member, const float &value) override;
      bool set2f(const std::string &member, const vec2f &value) override;
      bool set3f(const std::string &member, const vec3f &value) override;
      bool set1i(const std::string &member, const int &value) override;
      /*! @} */
      // ------------------------------------------------------------------

      MCGrid::SP buildMCs() override;

      /*! (re)computes the macro-cell ranges from the current filter state;
          buildMCs() only allocates and does the first pass */
      void refreshMCs();

      /*! recompacts the list of primitives the filter keeps, and bumps
          filterEpoch so anything derived from it knows to redo itself */
      void updateActiveList();

      /*! sums the enabled channels into the dense per-blob polynomial. Called
          when the channel mask changes - which is the only thing it depends
          on - so that sampling never has to look at a disabled channel. */
      void recontractChannels();

      /*! incremented every time the active list is recomputed. the sampler
          keys its bvh off this rather than off the active *count*: two
          different selections can easily contain the same number of
          primitives - symmetry-equivalent groups make that likely, not rare -
          and a count comparison would then skip a rebuild the contents
          needed, leaving hidden groups visible and shown ones missing. */
      int filterEpoch = 0;

      struct PLD {
        /*! the enabled channels summed per blob; see DD::polyDense */
        float    *polyDense = nullptr;
        size_t    denseCapacity = 0;
        /*! channel mask polyDense was built for; ~0u means never built */
        uint32_t  denseMask = 0u;
        bool      denseValid = false;
        /*! compacted indices of the primitives this filter keeps */
        uint32_t *activeIDs = nullptr;
        /*! per-primitive membership, in primitive order - the compacted list
            above comes out in atomic order and so cannot be compared */
        uint8_t  *kept      = nullptr;
        std::vector<uint8_t> prevKept;
        int       numActive = 0;
        size_t    capacity  = 0;
      };
      PLD *getPLD(Device *device);
      std::vector<PLD> perLogical;

      /*! bounds of the active primitives, in active-list order, for the
          cuBQL build */
      void computeElementBBs(Device *device, box3f *d_primBounds);

      /* no createIsoAccel(): this field is volume-only. the base class then
         returns an empty accel, so asking for an iso-surface on it is a no-op
         rather than an error. */
      VolumeAccel::SP createAccel(Volume *volume) override;

      /*! the accels built on this field. A filter change commits the *field*,
          not the volume, so nothing else would tell them that the cell ranges -
          and therefore their majorants - just moved. Weak, because the accel
          owns its volume and the volume owns us. */
      std::vector<std::weak_ptr<VolumeAccel>> accels;

      /*! shared across the volume and any iso-surfaces built on this field */
      std::shared_ptr<ParticleRBFSampler> sampler;

      struct {
        PODData::SP/*3f*/ centers   = 0;
        PODData::SP/*1f*/ coeffs    = 0;
        PODData::SP/*1f*/ gammas    = 0;
        PODData::SP/*1f*/ cutoffs   = 0;
        PODData::SP/*1ui*/monomials = 0;
        PODData::SP/*1ui*/groupIDs  = 0;
        PODData::SP/*1ui*/polyOffsets   = 0;
        PODData::SP/*1f*/ polyCoeffs    = 0;
        PODData::SP/*1ui*/polyMonomials = 0;
      } particle;
      struct {
        PODData::SP/*1ui8*/enabled = 0;
        PODData::SP/*1f*/  values  = 0;
        PODData::SP/*3f*/  directions = 0;
      } group;

      int      numParticles = 0;
      int      numGroups    = 0;
      /*! how many orbital channels each primitive carries a polynomial for */
      int      numChannels  = 1;
      /*! bit c enables channel c. all-ones by default. */
      uint32_t channelMask  = 0xffffffffu;
      int      vecMode      = VEC_MAGNITUDE;
      /*! highest monomial degree present; sizes the per-blob power table and
          the dense coefficient stride */
      int      polyMaxDegree = RBF_MAX_DEGREE;
      range1f filterValueRange = { 0.f, std::numeric_limits<float>::infinity() };
      int     mcGridSize   = DEFAULT_MC_GRID_SIZE;
      /*! how many probe points per axis the macro-cell range refinement uses;
          0 turns it off and leaves the conservative bound in place */
      int     mcSamplesPerAxis = 3;
      /*! how far past the sampled range each cell is padded, as a fraction of
          the sampled span. this is what buys back the features the probes
          missed, so it trades robustness against space skipping. */
      float   mcRangePad = 0.5f;
      /*! explicit world bounds; if non-empty this wins over the bounds
          derived from the primitives' supports (which bulge out past the
          simulation cell by a cutoff radius) */
      box3f   explicitBounds;
    };

#if RTC_DEVICE_CODE
    inline __rtc_device float rbf_powi(float x, int n)
    {
      float r = 1.f;
      while (n-- > 0) r *= x;
      return r;
    }

    /*! the largest |x^px y^py z^pz| attainable on the sphere |d|=1; the
        constrained maximum puts weight on each axis in proportion to its
        power. returns 1 for the isotropic case. */
    inline __rtc_device float rbf_monomialPeak(int px, int py, int pz)
    {
      const int n = px+py+pz;
      if (n == 0) return 1.f;
      const float rn = 1.f/float(n);
      float m = 1.f;
      if (px) m *= powf(px*rn,.5f*px);
      if (py) m *= powf(py*rn,.5f*py);
      if (pz) m *= powf(pz*rn,.5f*pz);
      return m;
    }

    /*! sum_{enabled c} sum_{terms} coeff * dx^px dy^py dz^pz, times the
        gaussian. The monomial powers are read per term, but the exponential -
        by far the expensive part - is evaluated once for the whole blob, which
        is the entire point of consolidating the angular block onto one
        (centre, gamma). */
    inline __rtc_device
    float ParticleRBFField::DD::evalPoly(int pid, vec3f d, float r2) const
    {
      /* Nested Horner: a polynomial in x whose coefficients are polynomials in
         y whose coefficients are polynomials in z. Every coefficient is
         consumed once, in storage order, by one fma - no power tables, no
         per-term multiplies, and 21 fewer registers held live. For degree 4
         that is 55 fma against 12 multiplies plus 105 ops.

         This is roughly neutral for frame rate - inside the intersection
         program the wider form's independent products hide latency about as
         well - but it halves the macro-cell refresh, which runs the same
         sampler ~1.6M times in a plain compute kernel, and that refresh is what
         a channel toggle waits on. */
      const float *co = polyDense + (size_t)pid*polyStride;
      const int D = polyMaxDegree;
      int slot = 0;
      float acc = 0.f;
      for (int kx=D;kx>=0;kx--) {
        const int M = D-kx;
        float accY = 0.f;
        for (int ky=M;ky>=0;ky--) {
          float accZ = 0.f;
          for (int kz=M-ky;kz>=0;kz--)
            accZ = accZ*d.z + co[slot++];
          accY = accY*d.y + accZ;
        }
        acc = acc*d.x + accY;
      }
      /* __expf: the hardware approximation. ~2 ulp against expf, against a
         field whose own coefficients are float and whose transfer function
         quantises to 8 bits - and there are ~132 of these per sample point. */
      return acc * __expf(-gammas[pid]*r2);
    }

    inline __rtc_device
    float ParticleRBFField::DD::evalParticle(int pid, vec3f P) const
    {
      const vec3f d  = P - centers[pid];
      const float r2 = dot(d,d);
      const float cut = cutoffs[pid];
      if (r2 >= cut*cut) return 0.f;
      if (groupIDs && !filter.passes(groupIDs[pid])) return 0.f;

      if (polyOffsets) {
        const float f = evalPoly(pid,d,r2);
        /* a scalar consumer of a vector field gets the requested component;
           the magnitude is not available here because it needs the *sum* over
           primitives, so the sampler handles VEC_MAGNITUDE itself */
        if (!directions) return f;
        const vec3f u = directions[groupIDs ? groupIDs[pid] : 0];
        switch (vecMode) {
        case VEC_X: return f*u.x;
        case VEC_Y: return f*u.y;
        case VEC_Z: return f*u.z;
        default:    return f;
        }
      }

      float v = coeffs[pid] * expf(-gammas[pid]*r2);
      if (monomials) {
        const uint32_t m = monomials[pid];
        if (m) {
          v *= rbf_powi(d.x,(m      )&0xff);
          v *= rbf_powi(d.y,(m >>  8)&0xff);
          v *= rbf_powi(d.z,(m >> 16)&0xff);
        }
      }
      return v;
    }

    inline __rtc_device
    vec3f ParticleRBFField::DD::evalParticleVec(int pid, vec3f P) const
    {
      const vec3f d  = P - centers[pid];
      const float r2 = dot(d,d);
      const float cut = cutoffs[pid];
      if (r2 >= cut*cut) return vec3f(0.f);
      if (groupIDs && !filter.passes(groupIDs[pid])) return vec3f(0.f);
      const float f = evalPoly(pid,d,r2);
      const vec3f u = directions[groupIDs ? groupIDs[pid] : 0];
      return vec3f(f*u.x,f*u.y,f*u.z);
    }

    inline __rtc_device
    box3f ParticleRBFField::DD::particleBounds(int pid) const
    {
      const vec3f c = centers[pid];
      const float r = cutoffs[pid];
      return box3f{ c-r, c+r };
    }

    inline __rtc_device
    bool ParticleRBFField::DD::signDefinite(int pid) const
    {
      /* a polynomial blob mixes many monomials, so nothing keeps it one-sided;
         |j| is non-negative, but that is a property of the *sum*, not of one
         primitive, so the majorant still has to allow both signs here */
      if (polyOffsets) return false;
      if (!monomials) return true;
      const uint32_t m = monomials[pid];
      return ((((m)&0xff) | ((m>>8)&0xff) | ((m>>16)&0xff)) & 1) == 0;
    }

    inline __rtc_device
    float ParticleRBFField::DD::particleBound(int pid, float rLo, float rHi) const
    {
      const float gamma = gammas[pid];
      const float cut   = cutoffs[pid];
      if (rLo >= cut) return 0.f;
      rHi = min(rHi,cut);

      if (polyOffsets) {
        /* sum each term's own maximum over [rLo,rHi]: a triangle inequality
           over the polynomial, so it bounds |f| however the terms cancel.
           Reads the recontracted coefficients, so it bounds exactly the
           channels currently enabled. */
        const float *co = polyDense + (size_t)pid*polyStride;
        const int D = polyMaxDegree;
        float bound = 0.f;
        int slot = 0;
        /* same traversal order as evalPoly, so 'slot' stays in step with it */
        for (int qx=D;qx>=0;qx--)
         for (int qy=D-qx;qy>=0;qy--)
          for (int qz=D-qx-qy;qz>=0;qz--,slot++) {
              const float c = co[slot];
              if (c == 0.f) continue;
              const int dg = qx+qy+qz;
              float rPeak = (dg == 0) ? 0.f : sqrtf(float(dg)/(2.f*gamma));
              rPeak = max(rLo,min(rHi,rPeak));
              bound += fabsf(c) * rbf_monomialPeak(qx,qy,qz)
                * rbf_powi(rPeak,dg) * expf(-gamma*rPeak*rPeak);
            }
        return bound;
      }

      int px = 0, py = 0, pz = 0;
      if (monomials) {
        const uint32_t m = monomials[pid];
        px = (m      )&0xff;
        py = (m >>  8)&0xff;
        pz = (m >> 16)&0xff;
      }
      const int n = px+py+pz;

      /* r^n*exp(-gamma r^2) is unimodal with its peak at sqrt(n/2gamma), so
         the max over [rLo,rHi] sits at the peak clamped into that interval */
      float rPeak = (n == 0) ? 0.f : sqrtf(float(n)/(2.f*gamma));
      rPeak = max(rLo,min(rHi,rPeak));

      const float ang = rbf_monomialPeak(px,py,pz) * rbf_powi(rPeak,n);
      return fabsf(coeffs[pid]) * ang * expf(-gamma*rPeak*rPeak);
    }
#endif
  }
}
