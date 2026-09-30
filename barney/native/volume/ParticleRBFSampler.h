// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/volume/ParticleRBFField.h"
#include "native/volume/MCAccelerator.h"
#include "native/common/CuBQL.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace BARNEY_NS {
  namespace native {

    struct ParticleRBFField;

    /*! samples a ParticleRBFField by range-querying a cuBQL bvh over the
        primitives' supports and summing every primitive that covers the query
        point. this is the software-traversal equivalent of an rt-core range
        query; it has to be software because barney evaluates volumes inside an
        intersection program, and nested tracing is not available there. */
    struct ParticleRBFSampler : public ScalarFieldSampler {
      enum { BVH_WIDTH = 4 };
      using bvh_t  = cuBQL::WideBVH<float,3,BVH_WIDTH>;
      using node_t = typename bvh_t::Node;

      struct DD : public ParticleRBFField::DD {
#if RTC_DEVICE_CODE
        inline __rtc_device float sample(vec3f P, bool dbg = false) const;
#endif
        bvh_t bvh;
      };
      DD getDD(Device *device);

      struct PLD {
        bvh_t bvh = { 0,0 };
        /*! which filterEpoch this bvh was built for; -1 means never built */
        int   builtEpoch = -1;
      };
      PLD *getPLD(Device *device);
      std::vector<PLD> perLogical;

      ParticleRBFSampler(ParticleRBFField *field);

      /*! builds the string that allows for properly matching optix device
          progs for this type */
      inline static std::string typeName() { return "ParticleRBF"; }

      void build() override;

      /*! replaces each non-empty macro-cell's conservative range with one
          measured by sampling the cell, clamped to stay inside the conservative
          bound. the conservative bound is a sum of per-primitive maxima, so on
          a signed field with cancellation it is loose by a factor of 5-17 here
          - loose enough that every occupied cell brackets any small iso-value
          and space skipping stops working entirely. */
      void refineMCRanges(MCGrid::SP mcGrid, int samplesPerAxis, float pad);

      ParticleRBFField *const field;
      const DevGroup::SP devices;
    };

#if RTC_DEVICE_CODE
    inline __rtc_device
    float ParticleRBFSampler::DD::sample(vec3f P, bool dbg) const
    {
      /* a filter that hides everything leaves no bvh at all; traversing it
         would be an illegal access, and zero is the right answer anyway */
      if (bvh.nodes == nullptr || numActive == 0) return 0.f;

      cuBQL::box3f box; box.lower = box.upper = (const cuBQL::vec3f &)P;

      /* |j| is the length of the *summed* current density, so the vector has
         to be accumulated across primitives and reduced once at the end.
         Reducing per primitive would give sum|j_AB|, which double-counts every
         place two bonds carry current in opposite directions - exactly the
         counterflow this field is meant to show. */
      if (directions && vecMode == ParticleRBFField::VEC_MAGNITUDE) {
        vec3f acc(0.f);
        auto vlambda = [&](const uint32_t idx) -> int {
          const vec3f v = evalParticleVec((int)activeIDs[idx],P);
          acc.x += v.x; acc.y += v.y; acc.z += v.z;
          return CUBQL_CONTINUE_TRAVERSAL;
        };
        cuBQL::fixedBoxQuery::forEachPrim(vlambda,bvh,box);
        return sqrtf(acc.x*acc.x + acc.y*acc.y + acc.z*acc.z);
      }

      float sum = 0.f;
      /* the bvh indexes the active list, not the primitive arrays */
      auto lambda = [&](const uint32_t idx) -> int {
        sum += evalParticle((int)activeIDs[idx],P);
        return CUBQL_CONTINUE_TRAVERSAL;
      };
      cuBQL::fixedBoxQuery::forEachPrim(lambda,bvh,box);
      /* zero is the right answer outside every support - the field really is
         empty there, so unlike the interpolating samplers we never return NAN */
      return sum;
    }
#endif
  }
}
