// SPDX-FileCopyrightText:
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier:
// Apache-2.0

#pragma once

#include "barney/nface/NFaceField.h"
#include "barney/common/CuBQL.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace BARNEY_NS {

  /*! a shared-face polyhedral ("nface") scalar field, with a CuBQL
      bvh sampler */
  struct NFaceCuBQLSampler : public ScalarFieldSampler {
    using bvh_t  = cuBQL::BinaryBVH<float,3>;
    using node_t = typename bvh_t::Node;

    struct DD : public NFaceField::DD {
      inline __rtc_device float sample(vec3f P, bool dbg = false) const;

      bvh_t bvh;
    };
    DD getDD(Device *device);

    /*! per-device data - parent stores the field, we just store the
      bvh nodes */
    struct PLD {
      bvh_t bvh = { 0,0,0,0 };
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;

    NFaceCuBQLSampler(NFaceField *mesh);

    /*! builds the string that allows for properly matching optix
      device progs for this type */
    inline static std::string typeName() { return "NFace_CuBQL"; }

    void build() override;

    NFaceField *const mesh;
    const DevGroup::SP devices;
  };

  inline __rtc_device
  float NFaceCuBQLSampler::DD::sample(vec3f P, bool dbg) const
  {
    typename bvh_t::box_t box; box.lower = box.upper = to_cubql(P);

    float retVal = NAN;
    auto lambda = [this,P,&retVal,dbg]
      (const uint32_t primID)
    {
      if (this->cellScalar(retVal,primID,P,dbg))
        return CUBQL_TERMINATE_TRAVERSAL;
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    cuBQL::fixedBoxQuery::forEachPrim(lambda,bvh,box);
    return retVal;
  }

} // ::BARNEY_NS
